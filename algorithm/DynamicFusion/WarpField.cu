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

	struct MeshWarper
	{
		const GpuMesh::PointType* vsrc;
		const GpuMesh::PointType* nsrc;
		cudaTextureObject_t knnTex;
		cudaTextureObject_t nodesDqVwTex;
		GpuMesh::PointType* vdst;
		GpuMesh::PointType* ndst;
		int num;

		Tbx::Quat_cu R;
		float3 t;

		float3 origion;
		float invVoxelSize;

		__device__ __forceinline__ void operator()(int tid)
		{
			float3 p = GpuMesh::from_point(vsrc[tid]);
			float3 n = GpuMesh::from_point(nsrc[tid]);

			Tbx::Dual_quat_cu dq_blend = WarpField::calc_dual_quat_blend_on_p(knnTex,
				nodesDqVwTex, p, origion, invVoxelSize);

			Tbx::Point3 dq_p = dq_blend.transform(Tbx::Point3(convert(p)));
			Tbx::Vec3 dq_n = dq_blend.rotate(convert(n));

			vdst[tid] = GpuMesh::to_point(convert(R.rotate(dq_p)) + t);
			ndst[tid] = GpuMesh::to_point(convert(R.rotate(dq_n)));
		}
	};

	__global__ void warp_mesh_kernel(MeshWarper warper)
	{
		unsigned int i = blockIdx.x * (blockDim.x << 3) + threadIdx.x;

#pragma unroll
		for (int k = 0; k < 8; k++)
		{
			if (i < warper.num)
			{
				warper(i);
			}
			i += blockDim.x;
		}
	}
	void WarpField::warp(GpuMesh& src, GpuMesh& dst)
	{
		if (src.num() == 0)
			return;

		dst.create(src.num());

		src.lockVertsNormals();
		dst.lockVertsNormals();

		MeshWarper warper;
		warper.t = convert(m_rigidTransform.get_translation());
		warper.R = Tbx::Quat_cu(m_rigidTransform);
		warper.knnTex = bindKnnFieldTexture();
		warper.nodesDqVwTex = bindNodesDqVwTexture();
		warper.vsrc = src.verts();
		warper.nsrc = src.normals();
		warper.vdst = dst.verts();
		warper.ndst = dst.normals();
		warper.num = src.num();
		warper.origion = m_volume->getOrigion();
		warper.invVoxelSize = 1.f / m_volume->getVoxelSize();

		dim3 block(512);
		dim3 grid(1, 1, 1);
		grid.x = divUp(dst.num(), block.x << 3);
		warp_mesh_kernel << <grid, block >> >(warper);
		cudaSafeCall(cudaGetLastError(), "warp");

		unBindKnnFieldTexture(warper.knnTex);
		unBindNodesDqVwTexture(warper.nodesDqVwTex);
		dst.unlockVertsNormals();
		src.unlockVertsNormals();
	}
#pragma endregion

#pragma region --init knn field
	__global__ void initKnnFieldKernel(cudaSurfaceObject_t knnSurf, int3 resolution)
	{
		int ix = blockDim.x*blockIdx.x + threadIdx.x;
		int iy = blockDim.y*blockIdx.y + threadIdx.y;
		int iz = blockDim.z*blockIdx.z + threadIdx.z;

		if (ix < resolution.x && iy < resolution.y && iz < resolution.z)
		{
			WarpField::KnnIdx idx = make_ushort4(WarpField::MaxNodeNum, WarpField::MaxNodeNum, 
				WarpField::MaxNodeNum, WarpField::MaxNodeNum);
			surf3Dwrite(idx, knnSurf, ix*sizeof(WarpField::KnnIdx), iy, iz);
		}
	}

	__global__ void initKnnFieldKernel1(WarpField::KnnIdx* knnPtr, int n)
	{
		int ix = blockDim.x*blockIdx.x + threadIdx.x;

		if (ix < n)
		{
			knnPtr[ix] = make_ushort4(WarpField::MaxNodeNum, WarpField::MaxNodeNum,
				WarpField::MaxNodeNum, WarpField::MaxNodeNum);
		}
	}

	void WarpField::initKnnField()
	{
		int3 res = m_volume->getResolution();
		dim3 block(32, 8, 2);
		dim3 grid(divUp(res.x, block.x),
			divUp(res.y, block.y),
			divUp(res.z, block.z));

		cudaSurfaceObject_t surf = bindKnnFieldSurface();
		initKnnFieldKernel << <grid, block >> >(surf, res);
		cudaSafeCall(cudaGetLastError(), "initKnnFieldKernel");
		unBindKnnFieldSurface(surf);

		dim3 block1(256);
		dim3 grid1(divUp(m_nodesGraph.size(), block1.x));
		initKnnFieldKernel1 << <grid, block >> >(m_nodesGraph.ptr(), m_nodesGraph.size());
		cudaSafeCall(cudaGetLastError(), "initKnnFieldKernel1");
	}
#pragma endregion

#pragma region --update nodes
	__device__ int newPoints_global_count = 0;
	__device__ int newPoints_output_count;
	__device__ unsigned int newPoints_blocks_done = 0;
	struct NewPointsCounter
	{
		enum
		{
			CTA_SIZE = 256,
			WARPS_COUNT = CTA_SIZE / Warp::WARP_SIZE
		};

		mutable int* out_keys;
		mutable float4* out_points;
		GpuMesh::PointType* input_points;
		cudaTextureObject_t knnTex;
		cudaTextureObject_t nodesDqVwTex;
		float4* nodesDqVw;

		int n;
		int step;
		float3 origion;
		int numNodes;

		// for volume index
		float vol_invVoxelSize;
		int3 vol_res;

		// for key generation
		float key_invGridSize;
		int3 key_gridRes;

		__device__ __forceinline__ void operator () () const
		{
			int tid = threadIdx.x + blockIdx.x * CTA_SIZE;

			if (__all(tid >= n))
				return;

			int warp_id = Warp::id();
			int lane_id = Warp::laneId();
			volatile __shared__ int warps_buffer[WARPS_COUNT];

			int flag = 0;
			int key = 0;
			float4 p4;
			if (tid < n)
			{
				float3 p = GpuMesh::from_point(input_points[tid*step]);
				p4 = GpuMesh::to_point(p, 1.f);

				// generating key
				float3 p1 = (p - origion)*key_invGridSize;
				int x = int(p1.x);
				int y = int(p1.y);
				int z = int(p1.z);

				key = (z*key_gridRes.y + y)*key_gridRes.x + x;

				// identify voxel
				p1 = (p - origion)*vol_invVoxelSize;
				x = int(p1.x);
				y = int(p1.y);
				z = int(p1.z);

				// assert knnIdx sorted, thus the 1st should be the nearest
				WarpField::KnnIdx knnIdx = make_ushort4(0,0,0,0);
				tex3D(&knnIdx, knnTex, x, y, z);

				if (knnIdx.x < numNodes)
				{
					float4 nearestVw = make_float4(0, 0, 0, 1);
					tex1Dfetch(&nearestVw, nodesDqVwTex, knnIdx.x * 3 + 2); // [q0-q1-vw] memory stored

					float3 nearestV = make_float3(nearestVw.x, nearestVw.y, nearestVw.z);

					// note .w store 1/radius
					float dif = dot(nearestV - p, nearestV - p) * (nearestVw.w * nearestVw.w);
					flag = (dif > 1.f);
				}
			}

			int total = __popc(__ballot(flag>0));

			if (total)
			{
				if (lane_id == 0)
				{
					int old = atomicAdd(&newPoints_global_count, total);
					warps_buffer[warp_id] = old;
				}

				int old_global_voxels_count = warps_buffer[warp_id];
				int offs = Warp::binaryExclScan(__ballot(flag>0));
				if (old_global_voxels_count + offs < n && flag)
				{
					out_keys[old_global_voxels_count + offs] = key;
					out_points[old_global_voxels_count + offs] = p4;
				}
			}// end if total

			if (Block::flattenedThreadId() == 0)
			{
				unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
				unsigned int value = atomicInc(&newPoints_blocks_done, total_blocks);

				//last block
				if (value == total_blocks - 1)
				{
					newPoints_output_count = newPoints_global_count;
					newPoints_blocks_done = 0;
					newPoints_global_count = 0;
				}
			}
		} /* operator () */
	};

	__global__ void get_newPoints_kernel(NewPointsCounter counter)
	{
		counter();
	}

	__global__ void pointToKey_kernel(
		const GpuMesh::PointType* points,
		int* key, float4* copypoints, int n, int step,
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

	struct NodesWriter
	{
		const float4* points_not_compact;
		const int* index;
		float4* nodesDqVw;
		float inv_weight_radius;
		int num;

		cudaTextureObject_t knnTex;
		cudaTextureObject_t nodesDqVwTex;
		float3 origion;
		float invVoxelSize;

		__device__ __forceinline__ void operator()(int threadId)
		{
			int idx = index[threadId];
			float4 p = points_not_compact[idx];
			float inv_w = 1.f / p.w;
			p.x *= inv_w;
			p.y *= inv_w;
			p.z *= inv_w;
			p.w = inv_weight_radius;
			nodesDqVw[threadId * 3 + 2] = p;

			Tbx::Dual_quat_cu dq_blend = WarpField::calc_dual_quat_blend_on_p(knnTex,
				nodesDqVwTex, make_float3(p.x, p.y, p.z), origion, invVoxelSize);
			unpack_dual_quat(dq_blend, nodesDqVw[threadId * 3], nodesDqVw[threadId * 3 + 1]);
		}
	};

	__global__ void write_nodes_kernel(NodesWriter nw)
	{
		int threadId = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadId < nw.num)
		{
			nw(threadId);
		}
	}

	void WarpField::insertNewNodes(GpuMesh& src)
	{
		// make a larger buffer to prevent allocation each time
		int step = m_param.warp_point_step_before_update_node;
		int num_points = src.num() / step;

		if (num_points == 0)
			return;

		if (num_points > m_current_point_buffer_size)
		{
			m_current_point_buffer_size = num_points * 1.5;
			m_meshPointsSorted.create(m_current_point_buffer_size);
			m_meshPointsKey.create(m_current_point_buffer_size);
			m_meshPointsFlags.create(m_current_point_buffer_size);
			m_tmpBuffer.create(m_current_point_buffer_size);

			cudaMemset(m_meshPointsSorted.ptr(), 0, m_meshPointsSorted.size()*m_meshPointsSorted.elem_size);
			cudaMemset(m_meshPointsKey.ptr(), 0, m_meshPointsKey.size()*m_meshPointsKey.elem_size);
			cudaMemset(m_meshPointsFlags.ptr(), 0, m_meshPointsFlags.size()*m_meshPointsFlags.elem_size);
			cudaMemset(m_tmpBuffer.ptr(), 0, m_tmpBuffer.size()*m_tmpBuffer.elem_size);
		}

		// reset symbols
		int zero_mem_symbol = 0;
		cudaMemcpyToSymbol(newPoints_global_count, &zero_mem_symbol, sizeof(int));
		cudaMemcpyToSymbol(newPoints_blocks_done, &zero_mem_symbol, sizeof(int));
		cudaMemcpyToSymbol(validVoxel_global_count, &zero_mem_symbol, sizeof(int));
		cudaMemcpyToSymbol(validVoxel_blocks_done, &zero_mem_symbol, sizeof(int));
		cudaSafeCall(cudaDeviceSynchronize(), "set zero: new point");

		// if 1st in, then collect all points
		if (m_lastNumNodes[0] == 0)
		{
			dim3 block(256);
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
		}
		// else, collect non-covered points
		else
		{
			src.lockVertsNormals();
			NewPointsCounter counter;
			counter.n = num_points;
			counter.step = step;
			counter.origion = m_volume->getOrigion();
			counter.key_gridRes = m_nodesGridSize;
			counter.key_invGridSize = 1.f / m_param.warp_radius_search_epsilon;
			counter.vol_invVoxelSize = 1.f / m_volume->getVoxelSize();
			counter.vol_res = m_volume->getResolution();

			counter.input_points = src.verts();
			counter.out_points = m_meshPointsSorted.ptr();
			counter.out_keys = m_meshPointsKey.ptr();
			counter.knnTex = bindKnnFieldTexture();
			counter.nodesDqVwTex = bindNodesDqVwTexture();
			counter.nodesDqVw = getNodesDqVwPtr(0);
			counter.numNodes = m_numNodes[0];

			dim3 block1(NewPointsCounter::CTA_SIZE);
			dim3 grid1(divUp(num_points, block1.x));
			get_newPoints_kernel << <grid1, block1 >> >(counter);
			cudaSafeCall(cudaGetLastError(), "get_newPoints_kernel");
			cudaSafeCall(cudaDeviceSynchronize(), "get_newPoints_kernel sync");

			cudaSafeCall(cudaMemcpyFromSymbol(&num_points, newPoints_output_count, 
				sizeof(int)), "get_newPoints_kernel memcpy from symbol");

			unBindKnnFieldTexture(counter.knnTex);
			unBindNodesDqVwTexture(counter.nodesDqVwTex);
			src.unlockVertsNormals();
		}// end else

		if (num_points == 0)
			return;

		// sort
		thrust_wrapper::sort_by_key(m_meshPointsKey.ptr(), m_meshPointsSorted.ptr(), num_points);

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
		{
			dim3 block1(ValidVoxelCounter::CTA_SIZE);
			dim3 grid1(divUp(num_points, block1.x));
			get_validVoxel_kernel << <grid1, block1 >> >(counter);
			cudaSafeCall(cudaGetLastError(), "get_validVoxel_kernel");
			cudaSafeCall(cudaDeviceSynchronize(), "get_validVoxel_kernel sync");
		}

		int num_after_compact = 0;
		cudaSafeCall(cudaMemcpyFromSymbol(&num_after_compact, 
			validVoxel_output_count, sizeof(int)), "copy voxel count from symbol");
		m_numNodes[0] = min(m_lastNumNodes[0] + num_after_compact, MaxNodeNum);
		if (num_after_compact + m_lastNumNodes[0] > MaxNodeNum)
			printf("warning: too many nodes %d vs %d\n", num_after_compact + m_lastNumNodes[0], MaxNodeNum);

		if (m_numNodes[0] > m_lastNumNodes[0])
		{
			dim3 block(256);
			dim3 grid(1, 1, 1);
			grid.x = divUp(m_numNodes[0] - m_lastNumNodes[0], block.x);

			NodesWriter nw;
			nw.points_not_compact = m_meshPointsSorted.ptr();
			nw.index = m_meshPointsFlags.ptr();
			nw.nodesDqVw = getNodesDqVwPtr(0) + m_lastNumNodes[0] * 3;
			nw.num = m_numNodes[0] - m_lastNumNodes[0];
			nw.inv_weight_radius = 1.f / m_param.warp_param_dw;
			nw.origion = m_volume->getOrigion();
			nw.invVoxelSize = 1.f / m_volume->getVoxelSize();
			nw.knnTex = bindKnnFieldTexture();
			nw.nodesDqVwTex = bindNodesDqVwTexture();

			write_nodes_kernel << <grid, block >> >(nw);
			cudaSafeCall(cudaGetLastError(), "write_nodes_kernel");

			unBindKnnFieldTexture(nw.knnTex);
			unBindNodesDqVwTexture(nw.nodesDqVwTex);
		}
	}
#pragma endregion

#pragma region --update ann field
	__global__ void seperate_xyz_nodes(const float4* nodesDqVw, 
		float* x, float* y, float* z, int n)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < n)
		{
			float4 dqVw = nodesDqVw[tid * 3 + 2];
			x[tid] = dqVw.x;
			y[tid] = dqVw.y;
			z[tid] = dqVw.z;
		}
	}

	__global__ void collect_aabb_box_kernel(float4* aabb_min, float4* aabb_max,
		const float* x, const float* y, const float* z, int n)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid == 0)
		{
			aabb_min[0] = make_float4(x[0], y[0], z[0], 0);
			aabb_max[0] = make_float4(x[n-1], y[n - 1], z[n - 1], 0);
		}
	}

	void WarpField::updateAnnField()
	{
		float3 origion = m_volume->getOrigion();
		int3 res = m_volume->getResolution();
		float vsz = m_volume->getVoxelSize();

		// if 1st frame, then perform whole-volume search, which is slow
		if (m_lastNumNodes[0] == 0)
		{
			m_nodeTree[0]->buildTree(m_nodesQuatTransVw.ptr() + 2, m_numNodes[0], 3);
			cudaSurfaceObject_t surf = bindKnnFieldSurface();
			m_nodeTree[0]->knnSearchGpu(surf, make_int3(0, 0, 0), res, origion, vsz, KnnK);
			unBindKnnFieldSurface(surf);
		}
		// else, collect voxels around the new added node and then perform sub-volume searching
		else
		{
			int nNewNodes = m_numNodes[0] - m_lastNumNodes[0];

			// 1st step, collect bounding box of new nodes to avoid additional computation
			float* xptr = m_tmpBuffer.ptr() + nNewNodes;
			float* yptr = xptr + nNewNodes;
			float* zptr = yptr + nNewNodes;
			if (nNewNodes)
			{
				dim3 block(32);
				dim3 grid(divUp(nNewNodes, block.x));
				seperate_xyz_nodes << <grid, block >> >(getNodesDqVwPtr(0) + m_lastNumNodes[0] * 3, 
					xptr, yptr, zptr, nNewNodes);
				cudaSafeCall(cudaGetLastError(), "seperate_xyz_nodes");
			}

			modergpu_wrapper::mergesort(xptr, nNewNodes);
			modergpu_wrapper::mergesort(yptr, nNewNodes);
			modergpu_wrapper::mergesort(zptr, nNewNodes);

			// bounding box info
			float4 box[2];
			{
				dim3 block(1);
				dim3 grid(1);
				collect_aabb_box_kernel << <grid, block >> >(
					m_meshPointsSorted.ptr(), m_meshPointsSorted.ptr() + 1, xptr, yptr, zptr, nNewNodes);
				cudaSafeCall(cudaGetLastError(), "collect_aabb_box_kernel");
				cudaSafeCall(cudaMemcpy(box, m_meshPointsSorted.ptr(), 2 * sizeof(float4), 
					cudaMemcpyDeviceToHost));
			}

			// convert to volume index
			int3 begin = make_int3((box[0].x - origion.x) / vsz, 
				(box[0].y - origion.y) / vsz, (box[0].z - origion.z) / vsz);
			int3 end = make_int3((box[1].x - origion.x) / vsz + 1,
				(box[1].y - origion.y) / vsz + 1, (box[1].z - origion.z) / vsz + 1);
			int ext = ceil(m_param.warp_param_dw / vsz);
			begin.x = min(res.x - 1, max(0, begin.x - ext));
			begin.y = min(res.y - 1, max(0, begin.y - ext));
			begin.z = min(res.z - 1, max(0, begin.z - ext));
			end.x = max(1, min(res.x, end.x + ext));
			end.y = max(1, min(res.y, end.y + ext));
			end.z = max(1, min(res.z, end.z + ext));

			// perform knn search on the sub volume
			m_nodeTree[0]->buildTree(m_nodesQuatTransVw.ptr() + 2, m_numNodes[0], 3);
			cudaSurfaceObject_t surf = bindKnnFieldSurface();
			m_nodeTree[0]->knnSearchGpu(surf, begin, end, origion, vsz, KnnK);
			//m_nodeTree[0]->knnSearchGpu(surf, make_int3(0,0,0), res, origion, vsz, KnnK);
			unBindKnnFieldSurface(surf);
		}
	}
#pragma endregion

#pragma region --update graph
	void WarpField::updateGraph(int level)
	{
		if (level == 0)
			throw std::exception("called an invalid level function\n");

		int num_points = m_numNodes[level - 1];

		if (num_points == 0)
		{
			m_numNodes[level] = 0;
			return;
		}

		// reset symbols
		int zero_mem_symbol = 0;
		cudaMemcpyToSymbol(newPoints_global_count, &zero_mem_symbol, sizeof(int));
		cudaMemcpyToSymbol(newPoints_blocks_done, &zero_mem_symbol, sizeof(int));
		cudaMemcpyToSymbol(validVoxel_global_count, &zero_mem_symbol, sizeof(int));
		cudaMemcpyToSymbol(validVoxel_blocks_done, &zero_mem_symbol, sizeof(int));
		cudaSafeCall(cudaDeviceSynchronize(), "set zero: new point");

		float radius = m_param.warp_radius_search_epsilon * pow(m_param.warp_radius_search_beta, level);

		{
			dim3 block(32);
			dim3 grid(1, 1, 1);
			grid.x = divUp(num_points, block.x << 3);

			// copy to new buffer and generate sort key
			pointToKey_kernel << <grid, block >> >(
				getNodesDqVwPtr(level-1)+2, m_meshPointsKey.ptr(), m_meshPointsSorted.ptr(),
				num_points, 3, 1.f / radius, m_volume->getOrigion(), m_nodesGridSize);
			cudaSafeCall(cudaGetLastError(), "pointToKey_kernel lv");
		}

		if (num_points == 0)
			return;

		// sort
		thrust_wrapper::sort_by_key(m_meshPointsKey.ptr(), m_meshPointsSorted.ptr(), num_points);

		// segment scan
		thrust_wrapper::inclusive_scan_by_key(m_meshPointsKey.ptr(),
			m_meshPointsSorted.ptr(), m_meshPointsSorted.ptr(), num_points);

		// compact
		ValidVoxelCounter counter;
		counter.counts = m_meshPointsFlags.ptr();
		counter.key_sorted = m_meshPointsKey.ptr();
		counter.n = num_points;
		counter.weight_thre = 1;
		counter.points_scaned = m_meshPointsSorted.ptr();
		if (num_points)
		{
			dim3 block1(ValidVoxelCounter::CTA_SIZE);
			dim3 grid1(divUp(num_points, block1.x));
			get_validVoxel_kernel << <grid1, block1 >> >(counter);
			cudaSafeCall(cudaGetLastError(), "get_validVoxel_kernel lv");
			cudaSafeCall(cudaDeviceSynchronize(), "get_validVoxel_kernel lv sync");
		}

		int num_after_compact = 0;
		cudaSafeCall(cudaMemcpyFromSymbol(&num_after_compact,
			validVoxel_output_count, sizeof(int)), "copy voxel count from symbol");
		m_numNodes[level] = min(num_after_compact, MaxNodeNum);
		if (num_after_compact > MaxNodeNum)
			printf("warning: too many nodes %d vs %d in level\n", 
			num_after_compact + m_lastNumNodes[0], MaxNodeNum, level);

		// write level nodes
		if (m_numNodes[level] > 0)
		{
			dim3 block(32);
			dim3 grid(1, 1, 1);
			grid.x = divUp(m_numNodes[level], block.x);


			NodesWriter nw;
			nw.points_not_compact = m_meshPointsSorted.ptr();
			nw.index = m_meshPointsFlags.ptr();
			nw.nodesDqVw = getNodesDqVwPtr(level);
			nw.num = m_numNodes[level];
			nw.inv_weight_radius = 1.f / (m_param.warp_param_dw*pow(m_param.warp_radius_search_beta, level));
			nw.origion = m_volume->getOrigion();
			nw.invVoxelSize = 1.f / m_volume->getVoxelSize();
			nw.knnTex = bindKnnFieldTexture();
			nw.nodesDqVwTex = bindNodesDqVwTexture();

			write_nodes_kernel << <grid, block >> >(nw);
			cudaSafeCall(cudaGetLastError(), "write_nodes_kernel");
		}

		// build graph
		m_nodeTree[level]->buildTree(getNodesDqVwPtr(level) + 2, m_numNodes[level], 3);

		m_nodeTree[level]->knnSearchGpu(getNodesDqVwPtr(level - 1) + 2, 3,
			(IdxType*)getNodesEdgesPtr(level - 1), nullptr, KnnK, getNumNodesInLevel(level - 1));
	}
#pragma endregion
}