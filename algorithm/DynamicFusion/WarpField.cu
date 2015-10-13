#include "WarpField.h"
#include "GpuMesh.h"
#include "device_utils.h"
#include "TsdfVolume.h"
#include "cudpp\thrust_wrapper.h"
#include "cudpp\ModerGpuWrapper.h"
#include "GpuKdTree.h"
namespace dfusion
{
	__device__ __host__ __forceinline__ WarpField::IdxType& knn_k(WarpField::KnnIdx& knn, int k)
	{
		return ((WarpField::IdxType*)(&knn))[k];
	}
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

	struct MapWarper
	{
		PtrStep<float4> vsrc;
		PtrStep<float4> nsrc;
		cudaTextureObject_t knnTex;
		cudaTextureObject_t nodesDqVwTex;
		PtrStep<float4> vdst;
		PtrStep<float4> ndst;
		int w;
		int h;

		Tbx::Quat_cu R;
		float3 t;

		float3 origion;
		float invVoxelSize;

		__device__ __forceinline__ void operator()(int x, int y)
		{
			float3 p = GpuMesh::from_point(vsrc(y,x));
			float3 n = GpuMesh::from_point(nsrc(y,x));

			Tbx::Dual_quat_cu dq_blend = WarpField::calc_dual_quat_blend_on_p(knnTex,
				nodesDqVwTex, p, origion, invVoxelSize);

			Tbx::Point3 dq_p = dq_blend.transform(Tbx::Point3(convert(p)));
			Tbx::Vec3 dq_n = dq_blend.rotate(convert(n));

			vdst(y, x) = GpuMesh::to_point(convert(R.rotate(dq_p)) + t);
			ndst(y, x) = GpuMesh::to_point(convert(R.rotate(dq_n)));
		}
	};

	__global__ void warp_map_kernel(MapWarper warper)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < warper.w && y < warper.h)
			warper(x, y);
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
		cudaSafeCall(cudaGetLastError(), "warp mesh");

		unBindKnnFieldTexture(warper.knnTex);
		unBindNodesDqVwTexture(warper.nodesDqVwTex);
		dst.unlockVertsNormals();
		src.unlockVertsNormals();
	}

	void WarpField::warp(const MapArr& srcVmap, const MapArr& srcNmap,
		MapArr& dstVmap, MapArr& dstNmap)
	{
		const int w = srcVmap.cols();
		const int h = srcNmap.rows();

		dstVmap.create(h, w);
		dstNmap.create(h, w);

		MapWarper warper;
		warper.t = convert(m_rigidTransform.get_translation());
		warper.R = Tbx::Quat_cu(m_rigidTransform);
		warper.knnTex = bindKnnFieldTexture();
		warper.nodesDqVwTex = bindNodesDqVwTexture();
		warper.vsrc = srcVmap;
		warper.nsrc = srcNmap;
		warper.vdst = dstVmap;
		warper.ndst = dstNmap;
		warper.w = w;
		warper.h = h;
		warper.origion = m_volume->getOrigion();
		warper.invVoxelSize = 1.f / m_volume->getVoxelSize();

		dim3 block(32, 8);
		dim3 grid(divUp(w, block.x), divUp(h, block.y), 1);
		warp_map_kernel << <grid, block >> >(warper);
		cudaSafeCall(cudaGetLastError(), "warp map");

		unBindKnnFieldTexture(warper.knnTex);
		unBindNodesDqVwTexture(warper.nodesDqVwTex);
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

#if 0
			if (isnan(dq_blend.get_dual_part().w()))
			{
				printf("kernel: %d %f %f %f %f %f\n", threadId, p.x, p.y, p.z, p.w, inv_w);
				printf("dqb: %f %f %f %f, %f %f %f %f\n", 
					dq_blend.get_non_dual_part().w(), dq_blend.get_non_dual_part().i(), 
					dq_blend.get_non_dual_part().j(), dq_blend.get_non_dual_part().k(), 
					dq_blend.get_dual_part().w(), dq_blend.get_dual_part().i(),
					dq_blend.get_dual_part().j(), dq_blend.get_dual_part().k());
				printf("origion: %f %f %f; ivsz: %f\n", origion.x, origion.y, origion.z, invVoxelSize);

				Tbx::Dual_quat_cu dq_blend1(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
				float3 p1 = (make_float3(p.x, p.y, p.z) - origion)*invVoxelSize;
				int x = int(p1.x);
				int y = int(p1.y);
				int z = int(p1.z);
				WarpField::KnnIdx knnIdx = make_ushort4(0, 0, 0, 0);
				tex3D(&knnIdx, knnTex, x, y, z);

				printf("knnIdx: (%d, %d, %d) -> (%d %d %d %d)\n", 
					x,y,z,knnIdx.x, knnIdx.y, knnIdx.z, knnIdx.w);

				if (knn_k(knnIdx, 0) >= WarpField::MaxNodeNum)
				{
					dq_blend = Tbx::Dual_quat_cu::identity();
				}
				else
				{
					Tbx::Dual_quat_cu dq0;
					for (int k = 0; k < WarpField::KnnK; k++)
					{
						if (knn_k(knnIdx, k) < WarpField::MaxNodeNum)
						{
							WarpField::IdxType nn3 = knn_k(knnIdx, k) * 3;
							float4 q0, q1, vw;
							tex1Dfetch(&q0, nodesDqVwTex, nn3 + 0);
							tex1Dfetch(&q1, nodesDqVwTex, nn3 + 1);
							tex1Dfetch(&vw, nodesDqVwTex, nn3 + 2);
							// note: we store 1.f/radius in vw.w
							float w = __expf(-norm2(make_float3(vw.x - p.x, vw.y - p.y,
								vw.z - p.z)) * 2 * (vw.w*vw.w));
							Tbx::Dual_quat_cu dq = pack_dual_quat(q0, q1);
							if (k == 0)
								dq0 = dq;
							else
							{
								if (dq0.get_non_dual_part().dot(dq.get_non_dual_part()) < 0)
									w = -w;
							}
							printf("dq(%d, %f): %f %f %f %f, %f %f %f %f\n", k, w,
								dq.get_non_dual_part().w(), dq.get_non_dual_part().i(), 
								dq.get_non_dual_part().j(), dq.get_non_dual_part().k(),
								dq.get_dual_part().w(), dq.get_dual_part().i(), 
								dq.get_dual_part().j(), dq.get_dual_part().k());
							dq_blend1 = dq_blend1 + dq*w;
						}
					}
					printf("bld: %f %f %f %f, %f %f %f %f\n", dq_blend1.get_non_dual_part().w(),
						dq_blend1.get_non_dual_part().i(), dq_blend1.get_non_dual_part().j(), 
						dq_blend1.get_non_dual_part().k(),
						dq_blend1.get_dual_part().w(), dq_blend1.get_dual_part().i(), 
						dq_blend1.get_dual_part().j(),
						dq_blend1.get_dual_part().k());
					dq_blend1.normalize();
				}
			}
#endif

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

	__global__ void bruteforce_updateKnn_kernel(cudaTextureObject_t nodesDqVwTex,
		cudaSurfaceObject_t knnSurf, int3 res, int newNodesBegin, int newNodesEnd,
		float3 origion, float voxelSize)
	{
		int x = threadIdx.x + blockIdx.x*blockDim.x;
		int y = threadIdx.y + blockIdx.y*blockDim.y;
		int z = threadIdx.z + blockIdx.z*blockDim.z;

		if (x < res.x && y < res.y && z < res.z)
		{
			// compute all 4 dists stored
			WarpField::KnnIdx knn;
			surf3Dread(&knn, knnSurf, x*sizeof(WarpField::KnnIdx), y, z);
			float3 voxelPos = origion + voxelSize*make_float3(x, y, z);
			float oldDists2[WarpField::KnnK];
			for (int k = 0; k < WarpField::KnnK; k++)
			{
				float4 p;
				tex1Dfetch(&p, nodesDqVwTex, knn_k(knn, k)*3 + 2);
				oldDists2[k] = norm2(make_float3(p.x, p.y, p.z) - voxelPos);
			}
			
			// update new nodes
			for (int iNode = newNodesBegin; iNode < newNodesEnd; iNode++)
			{
				float4 p;
				tex1Dfetch(&p, nodesDqVwTex, iNode * 3 + 2);
				float newDist2 = norm2(make_float3(p.x, p.y, p.z) - voxelPos);

				// we swap the farest nodes out
				// note that the knn is kept sorted
				int swapPos = WarpField::KnnK;
				for (int k = 0; k < WarpField::KnnK; k++)
				{
					if (newDist2 < oldDists2[k])
					{
						swapPos = k;
						break;
					}
				}

				if (swapPos < WarpField::KnnK)
				{
					WarpField::KnnIdx newKnn = knn;
					knn_k(newKnn, swapPos) = iNode;
					for (int k = swapPos + 1; k < WarpField::KnnK; k++)
						knn_k(newKnn, k) = knn_k(knn, k - 1);
					surf3Dwrite(newKnn, knnSurf, x*sizeof(WarpField::KnnIdx), y, z);
				}
			}// end for iNode
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
#if 0
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
#else
			//tranverse each voxel to update
			if (nNewNodes > 0)
			{
				int3 res = m_volume->getResolution();
				float3 origion = m_volume->getOrigion();
				float vsz = m_volume->getVoxelSize();
				dim3 block(32, 8, 2);
				dim3 grid(divUp(res.x, block.x),
					divUp(res.y, block.y),
					divUp(res.z, block.z));

				cudaSurfaceObject_t surf = bindKnnFieldSurface();
				cudaTextureObject_t tex = bindNodesDqVwTexture();
				bruteforce_updateKnn_kernel << <grid, block >> >(
					tex, surf, res, m_lastNumNodes[0], m_numNodes[0], origion, vsz);
				cudaSafeCall(cudaGetLastError(), "bruteforce_updateKnn_kernel");
				unBindNodesDqVwTexture(tex);
				unBindKnnFieldSurface(surf);
			}
#endif
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
			nw.inv_weight_radius = 1.f / (m_param.warp_param_dw*pow(m_param.warp_param_dw_lvup_scale, level));
			nw.origion = m_volume->getOrigion();
			nw.invVoxelSize = 1.f / m_volume->getVoxelSize();
			nw.knnTex = bindKnnFieldTexture();
			nw.nodesDqVwTex = bindNodesDqVwTexture();

			write_nodes_kernel << <grid, block >> >(nw);
			cudaSafeCall(cudaGetLastError(), "write_nodes_kernel");
		}

		// build graph
		if (m_numNodes[level] > 0)
		{
			m_nodeTree[level]->buildTree(getNodesDqVwPtr(level) + 2, m_numNodes[level], 3);

			m_nodeTree[level]->knnSearchGpu(getNodesDqVwPtr(level - 1) + 2, 3,
				(IdxType*)getNodesEdgesPtr(level - 1), nullptr, KnnK, getNumNodesInLevel(level - 1));
		}
	}
#pragma endregion

#pragma region --extract_for_vmap
	struct IdxContainter
	{
		int id[WarpField::GraphLevelNum+1];
		__device__ __host__ int& operator [](int i)
		{
			return id[i];
		}
	};


	__global__ void extract_knn_for_vmap_kernel(PtrStepSz<float4> vmap, PtrStepSz<WarpField::KnnIdx> vmapKnn,
		float3 origion, float invVoxelSize, cudaTextureObject_t knnTex, IdxContainter ic)
	{
		int u = blockIdx.x * blockDim.x + threadIdx.x;
		int v = blockIdx.y * blockDim.y + threadIdx.y;

		if (u < vmap.cols && v < vmap.rows)
		{
			float3 p = GpuMesh::from_point(vmap(v, u));
			WarpField::KnnIdx knnIdx = make_ushort4(ic[WarpField::GraphLevelNum], 
				ic[WarpField::GraphLevelNum], ic[WarpField::GraphLevelNum], ic[WarpField::GraphLevelNum]);

			if (!isnan(p.x))
			{
				float3 p1 = (p - origion)*invVoxelSize;
				int x = int(p1.x);
				int y = int(p1.y);
				int z = int(p1.z);
				tex3D(&knnIdx, knnTex, x, y, z);
				for (int k = 0; k < WarpField::KnnK; k++)
				if (knn_k(knnIdx, k) >= WarpField::MaxNodeNum)
					knn_k(knnIdx, k) = ic[WarpField::GraphLevelNum];
			}

			vmapKnn(v, u) = knnIdx;
		}
	}

	void WarpField::extract_knn_for_vmap(const MapArr& vmap, DeviceArray2D<KnnIdx>& vmapKnn)const
	{
		IdxContainter ic;
		ic[0] = 0;
		for (int k = 0; k < GraphLevelNum; k++)
			ic[k + 1] = ic[k] + m_numNodes[k];

		vmapKnn.create(vmap.rows(), vmap.cols());

		dim3 block(32, 8);
		dim3 grid(divUp(vmap.cols(), block.x), divUp(vmap.rows(), block.y));

		cudaTextureObject_t knnTex = bindKnnFieldTexture();
		extract_knn_for_vmap_kernel << <grid, block >> >(vmap, vmapKnn, m_volume->getOrigion(),
			1.f / m_volume->getVoxelSize(), knnTex, ic);
		cudaSafeCall(cudaGetLastError(), "extract_knn_for_vmap_kernel");
		unBindKnnFieldTexture(knnTex);
	}

	__global__ void extract_nodes_info_kernel(const float4* nodesDqVw, float* twist, float4* vw,
		const WarpField::KnnIdx* nodesKnnIn, WarpField::KnnIdx* nodesKnnOut, IdxContainter ic)
	{
		int iout = blockIdx.x * blockDim.x + threadIdx.x;
		if (iout >= ic[WarpField::GraphLevelNum])
			return;

		int level = 0;
		for (int k = 0; k < WarpField::GraphLevelNum; k++)
		if (iout >= ic[k] && iout < ic[k + 1])
		{
			level = k;
			break;
		}

		int iin = level*WarpField::MaxNodeNum + iout - ic[level];

		// write twist
		Tbx::Dual_quat_cu dq = pack_dual_quat(nodesDqVw[iin * 3], nodesDqVw[iin * 3 + 1]);
		Tbx::Vec3 r, t;
		dq.to_twist(r, t);
		twist[iout * 6 + 0] = r.x;
		twist[iout * 6 + 1] = r.y;
		twist[iout * 6 + 2] = r.z;
		twist[iout * 6 + 3] = t.x;
		twist[iout * 6 + 4] = t.y;
		twist[iout * 6 + 5] = t.z;
		vw[iout] = nodesDqVw[iin * 3 + 2];

		// write knn
		WarpField::KnnIdx kid = nodesKnnIn[iin];
		for (int k = 0; k < WarpField::KnnK; k++)
		{
			knn_k(kid, k) = (knn_k(kid, k) < ic[level + 1] - ic[level] ? 
				knn_k(kid, k) + ic[level + 1] : ic[WarpField::GraphLevelNum]);
		}
		nodesKnnOut[iout] = kid;
	}

	void WarpField::extract_nodes_info(DeviceArray<KnnIdx>& nodesKnn, DeviceArray<float>& twist,
		DeviceArray<float4>& vw)const
	{
		IdxContainter ic;
		ic[0] = 0;
		for (int k = 0; k < GraphLevelNum; k++)
			ic[k + 1] = ic[k] + m_numNodes[k];

		if (ic[GraphLevelNum] == 0)
			return;

		nodesKnn.create(ic[GraphLevelNum]);
		twist.create(ic[GraphLevelNum] * 6);
		vw.create(ic[GraphLevelNum]);

		extract_nodes_info_no_allocation(nodesKnn, twist, vw);
	}

	void WarpField::extract_nodes_info_no_allocation(
		DeviceArray<KnnIdx>& nodesKnn,
		DeviceArray<float>& twist,
		DeviceArray<float4>& vw)const
	{
		IdxContainter ic;
		ic[0] = 0;
		for (int k = 0; k < GraphLevelNum; k++)
			ic[k + 1] = ic[k] + m_numNodes[k];

		if (ic[GraphLevelNum] == 0)
			return;

		dim3 block(256);
		dim3 grid(divUp(ic[GraphLevelNum], block.x));

		extract_nodes_info_kernel << <grid, block >> >(getNodesDqVwPtr(0),
			twist.ptr(), vw.ptr(), getNodesEdgesPtr(0), nodesKnn.ptr(), ic);
		cudaSafeCall(cudaGetLastError(), "extract_nodes_info_kernel");
	}

	__global__ void update_nodes_via_twist_kernel(float4* nodesDqVw, const float* twist,
		IdxContainter ic)
	{
		int iout = blockIdx.x * blockDim.x + threadIdx.x;
		if (iout >= ic[WarpField::GraphLevelNum])
			return;

		int level = 0;
		for (int k = 0; k < WarpField::GraphLevelNum; k++)
		if (iout >= ic[k] && iout < ic[k + 1])
		{
			level = k;
			break;
		}

		int iin = level*WarpField::MaxNodeNum + iout - ic[level];

		// write twist
		Tbx::Vec3 r, t;
		r.x = twist[iout * 6 + 0];
		r.y = twist[iout * 6 + 1];
		r.z = twist[iout * 6 + 2];
		t.x = twist[iout * 6 + 3];
		t.y = twist[iout * 6 + 4];
		t.z = twist[iout * 6 + 5];
		Tbx::Dual_quat_cu dq;
		dq.from_twist(r, t);
		unpack_dual_quat(dq, nodesDqVw[iin * 3], nodesDqVw[iin * 3 + 1]);
	}

	void WarpField::update_nodes_via_twist(const DeviceArray<float>& twist)
	{
		IdxContainter ic;
		ic[0] = 0;
		for (int k = 0; k < GraphLevelNum; k++)
			ic[k + 1] = ic[k] + m_numNodes[k];

		if (twist.size() < ic[GraphLevelNum]*6)
			throw std::exception("size not matched in WarpField::update_nodes_via_twist()");

		dim3 block(256);
		dim3 grid(divUp(ic[GraphLevelNum], block.x));

		update_nodes_via_twist_kernel << <grid, block >> >(getNodesDqVwPtr(0),
			twist.ptr(), ic);
		cudaSafeCall(cudaGetLastError(), "update_nodes_via_twist");
	}
#pragma endregion

#pragma region --getKnnAt

	__global__ void getKnnAtKernel(WarpField::KnnIdx* data, int3 p, cudaTextureObject_t tex)
	{
		tex3D(&data[0], tex, p.x, p.y, p.z);
	}

	WarpField::KnnIdx WarpField::getKnnAt(float3 volumePos)const
	{
		if (m_volume == nullptr)
			throw std::exception("WarpField::getKnnAt(): null pointer");
		float3 ori = m_volume->getOrigion();
		float vsz = m_volume->getVoxelSize();
		float3 p = (volumePos - ori) / vsz;
		return getKnnAt(make_int3(p.x, p.y, p.z));
	}
	WarpField::KnnIdx WarpField::getKnnAt(int3 gridXYZ)const
	{
		if (m_volume == nullptr)
			throw std::exception("WarpField::getKnnAt(): null pointer");
		int3 res = m_volume->getResolution();
		int x = gridXYZ.x, y = gridXYZ.y, z = gridXYZ.z;
		if (x < 0 || y < 0 || z < 0 || x >= res.x || y >= res.y || z >= res.z)
			return make_ushort4(MaxNodeNum, MaxNodeNum, MaxNodeNum, MaxNodeNum);
		static DeviceArray<KnnIdx> knn;
		knn.create(1);

		cudaTextureObject_t tex = bindKnnFieldTexture();
		getKnnAtKernel << <dim3(1), dim3(1) >> >(knn.ptr(), gridXYZ, tex);
		cudaSafeCall(cudaGetLastError(), "WarpField::getKnnAtKernel");
		unBindKnnFieldTexture(tex);

		WarpField::KnnIdx host;
		cudaSafeCall(cudaMemcpy(&host, knn.ptr(), sizeof(KnnIdx), cudaMemcpyDeviceToHost),
			"WarpField::getKnnAtKernel, post copy");
		return host;
	}
#pragma endregion
}