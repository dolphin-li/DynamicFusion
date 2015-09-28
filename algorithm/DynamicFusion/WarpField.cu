#include "WarpField.h"
#include "GpuMesh.h"
#include "device_utils.h"
#include "TsdfVolume.h"
#include "cudpp\thrust_wrapper.h"
#include "cudpp\ModerGpuWrapper.h"
#include "GpuKdTree.h"
namespace dfusion
{
#pragma region --warpmesh
	__global__ void warp_mesh_kernel(const  GpuMesh::PointType*__restrict__ vsrc,
		const GpuMesh::PointType* nsrc,
		GpuMesh::PointType* vdst, GpuMesh::PointType* ndst, Tbx::Quat_cu R, float3 t, int n)
	{
		unsigned int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		unsigned int i = __mul24(blockId, blockDim.x << 3) + threadIdx.x;

#pragma unroll
		for (int k = 0; k < 8; k++)
		{
			if (i < n)
			{
				vdst[i] = GpuMesh::to_point(convert(R.rotate(convert(GpuMesh::from_point(vsrc[i]))))+t);
				ndst[i] = GpuMesh::to_point(convert(R.rotate(convert(GpuMesh::from_point(nsrc[i])))));
			}
			i += blockDim.x;
		}
	}
	void WarpField::warp(GpuMesh& src, GpuMesh& dst)
	{
		dst.create(src.num());

		dim3 block(512);
		dim3 grid(1, 1, 1);
		grid.x = divUp(dst.num(), block.x<<3);

		//Mat33 R = convert(m_rigidTransform.get_mat3());
		float3 t = convert(m_rigidTransform.get_translation());
		Tbx::Quat_cu q(m_rigidTransform);

		src.lockVertsNormals();
		dst.lockVertsNormals();

		warp_mesh_kernel << <grid, block >> >(src.verts(), src.normals(), 
			dst.verts(), dst.normals(), q, t, src.num());
		cudaSafeCall(cudaGetLastError(), "warp");

		dst.unlockVertsNormals();
		src.unlockVertsNormals();
	}
#pragma endregion

#pragma region --update nodes

	__global__ void pointToKey_kernel(
		const GpuMesh::PointType* points, int* key,
		float4* copypoints, int n, int step,
		float invGridSize, float3 origion, int3 gridRes)
	{
		unsigned int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		unsigned int threadId = __mul24(blockId, blockDim.x << 3) + threadIdx.x;

#pragma unroll
		for (int k = 0; k < 8; k++, threadId += blockDim.x)
		{
			if (threadId < n)
			{
				float3 p = GpuMesh::from_point(points[threadId*step]);
				float3 p1 = (p- origion)*invGridSize;
				int x = int(p1.x);
				int y = int(p1.y);
				int z = int(p1.z);
				key[threadId] = (z*gridRes.y + y)*gridRes.x + x;
				copypoints[threadId] = GpuMesh::to_point(p, 1.f);
			}
		}
	}



	__device__ int validVoxel_global_count = 0;
	__device__ int validVoxel_output_count;
	__device__ unsigned int validVoxel_blocks_done = 0;
	struct ValidVoxelCounter
	{
		enum
		{
			CTA_SIZE = 256,
			WARPS_COUNT = CTA_SIZE / Warp::WARP_SIZE
		};

		mutable int* key_sorted;
		mutable int* counts;
		const float4* points_scaned;
		float weight_thre;
		int n;

		__device__ __forceinline__ void operator () () const
		{
			int tid = threadIdx.x + blockIdx.x * CTA_SIZE;

			if (__all(tid >= n))
				return;

			int warp_id = Warp::id();
			int lane_id = Warp::laneId();
			volatile __shared__ int warps_buffer[WARPS_COUNT];

			int flag = 0;
			if (tid < n)
				flag = (points_scaned[tid].w > weight_thre) &&
				(key_sorted[tid] != key_sorted[tid + 1] || tid == n - 1);
			int total = __popc(__ballot(flag>0));

			if (total)
			{
				if (lane_id == 0)
				{
					int old = atomicAdd(&validVoxel_global_count, total);
					warps_buffer[warp_id] = old;
				}

				int old_global_voxels_count = warps_buffer[warp_id];
				int offs = Warp::binaryExclScan(__ballot(flag>0));
				if (old_global_voxels_count + offs < n && flag)
					counts[old_global_voxels_count + offs] = tid;
			}// end if total

			if (Block::flattenedThreadId() == 0)
			{
				unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
				unsigned int value = atomicInc(&validVoxel_blocks_done, total_blocks);

				//last block
				if (value == total_blocks - 1)
				{
					validVoxel_output_count = validVoxel_global_count;
					validVoxel_blocks_done = 0;
					validVoxel_global_count = 0;
				}
			}
		} /* operator () */
	};

	__global__ void get_validVoxel_kernel(ValidVoxelCounter counter)
	{
		counter();
	}

	__global__ void write_nodes_kernel(const float4* points_not_compact, 
		const int* index, WarpNode* nodes, float4* nodesVW, float weight_radius, int n)
	{
		unsigned int blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		unsigned int threadId = __mul24(blockId, blockDim.x << 3) + threadIdx.x;

#pragma unroll
		for (int k = 0; k < 8; k++, threadId += blockDim.x)
		{
			if (threadId < n)
			{
				WarpNode node;
				float4 p = points_not_compact[index[threadId]];
				float inv_w = 1.f / p.w;
				node.v_w.x = p.x * inv_w;
				node.v_w.y = p.y * inv_w;
				node.v_w.z = p.z * inv_w;
				node.v_w.w = weight_radius;
				node.dq = Tbx::Dual_quat_cu::identity();
				nodes[threadId] = node;
				nodesVW[threadId] = node.v_w;
			}
		}
	}

	void WarpField::insertNewNodes(GpuMesh& src)
	{
		// make a larger buffer to prevent allocation each time
		int step = m_param.warp_point_step_before_update_node;
		int num_points = src.num() / step;
		if (num_points > m_current_point_buffer_size)
		{
			m_current_point_buffer_size = num_points * 1.5;
			m_meshPointsSorted.create(m_current_point_buffer_size);
			m_meshPointsKey.create(m_current_point_buffer_size);
			m_meshPointsFlags.create(m_current_point_buffer_size);
		}

		dim3 block(512);
		dim3 grid(1, 1, 1);
		grid.x = divUp(num_points, block.x << 3);

		// copy to new buffer and generate sort key
		src.lockVertsNormals();	
		pointToKey_kernel << <grid, block >> >(
			src.verts(), m_meshPointsKey.ptr(), m_meshPointsSorted.ptr(),
			num_points, step, 1.f / m_param.warp_radius_search_epsilon,
			m_volume->getOrigion(), m_nodesGridSize);
		cudaSafeCall(cudaGetLastError(), "pointToKey_kernel");
		src.unlockVertsNormals();

		// sort
		thrust_wrapper::sort_by_key(m_meshPointsKey.ptr(), m_meshPointsSorted.ptr(), num_points);
		//modergpu_wrapper::mergesort_by_key(m_meshPointsKey.ptr(), m_meshPointsSorted.ptr(), num_points);

		// segment scan
		thrust_wrapper::inclusive_scan_by_key(m_meshPointsKey.ptr(), 
			m_meshPointsSorted.ptr(), m_meshPointsSorted.ptr(), num_points);
		
		// compact
		ValidVoxelCounter counter;
		counter.counts = m_meshPointsFlags.ptr();
		counter.key_sorted = m_meshPointsKey.ptr();
		counter.n = num_points;
		counter.weight_thre = m_param.warp_valid_point_num_each_node;
		counter.points_scaned = m_meshPointsSorted.ptr();

		dim3 block1(ValidVoxelCounter::CTA_SIZE);
		dim3 grid1(divUp(num_points, block1.x));
		get_validVoxel_kernel << <grid1, block1 >> >(counter);
		cudaSafeCall(cudaGetLastError(), "get_validVoxel_kernel");
		cudaSafeCall(cudaDeviceSynchronize());

		int num_after_compact = 0;
		cudaSafeCall(cudaMemcpyFromSymbol(&num_after_compact, validVoxel_output_count, sizeof(int)));
		m_numNodes[0] = min(num_after_compact, MaxNodeNum);
		if (num_after_compact > MaxNodeNum)
			printf("warning: too many nodes %d vs %d\n", num_after_compact, MaxNodeNum);
		write_nodes_kernel << <grid, block >> >(
			m_meshPointsSorted.ptr(), m_meshPointsFlags.ptr(), getNodesPtr(0), getNodesVwPtr(0),
			m_param.warp_param_dw, m_numNodes[0]);
		cudaSafeCall(cudaGetLastError(), "write_nodes_kernel");
	}
#pragma endregion

#pragma region --update ann field
	void WarpField::updateAnnField()
	{
		m_nodeTree->buildTree(m_nodesVW.ptr(), m_numNodes[0]);
	}
#pragma endregion

#pragma region --update graph
	void WarpField::updateGraph(int level)
	{

	}
#pragma endregion
}