#include "GpuMesh.h"
#include "WarpField.h"
#include "TsdfVolume.h"
#include <helper_math.h>
#include "device_utils.h"
namespace dfusion
{
	__device__ __forceinline__ float3 read_float3_4(float4 a)
	{
		return make_float3(a.x, a.y, a.z);
	}

	__device__ __forceinline__ PixelRGBA copy_uchar4_pixelRGBA(uchar4 a)
	{
		PixelRGBA p;
		p.r = a.x;
		p.g = a.y;
		p.b = a.z;
		p.a = a.w;
		return p;
	}

	__device__ __forceinline__ PixelRGBA copy_float4_pixelRGBA(float4 a)
	{
		PixelRGBA p;
		p.r = a.x * 255;
		p.g = a.y * 255;
		p.b = a.z * 255;
		p.a = a.w * 255;
		return p;
	}

	__global__ void copy_invert_y_kernel(PtrStepSz<float4> gldata,
		PtrStepSz<PixelRGBA> img)
	{
		int u = threadIdx.x + blockIdx.x * blockDim.x;
		int v = threadIdx.y + blockIdx.y * blockDim.y;

		if (u >= img.cols || v >= img.rows)
			return;

		img(v, u) = copy_float4_pixelRGBA(gldata(img.rows-1-v, u));
	}

	void GpuMesh::copy_invert_y(const float4* gldata, ColorMap& img)
	{
		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(img.cols(), block.x);
		grid.y = divUp(img.rows(), block.y);

		PtrStepSz<float4> gldataptr;
		gldataptr.data = (float4*)gldata;
		gldataptr.rows = img.rows();
		gldataptr.cols = img.cols();
		gldataptr.step = img.cols()*sizeof(float4);

		copy_invert_y_kernel << <grid, block >> >(gldataptr, img);
		cudaSafeCall(cudaGetLastError(), "GpuMesh::copy_invert_y");
	}

	__global__ void copy_gldepth_to_depthmap_kernel(PtrStepSz<float4> gldata,
		PtrStepSz<depthtype> img, float s1, float s2, float camNear)
	{
		int u = threadIdx.x + blockIdx.x * blockDim.x;
		int v = threadIdx.y + blockIdx.y * blockDim.y;

		if (u >= img.cols || v >= img.rows)
			return;

		float4 p = gldata(img.rows - 1 - v, u);
		float val = p.x;
		val = s1 / (2 * val - 1 + s2) * 1000.f;
		if (val <= camNear*1000.f)
			val = 0;
		img(v, u) = val;
	}

	void GpuMesh::copy_gldepth_to_depthmap(const float4* gldata, DepthMap& depth, 
		float s1, float s2, float camNear)
	{
		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(depth.cols(), block.x);
		grid.y = divUp(depth.rows(), block.y);

		PtrStepSz<float4> gldataptr;
		gldataptr.data = (float4*)gldata;
		gldataptr.rows = depth.rows();
		gldataptr.cols = depth.cols();
		gldataptr.step = depth.cols()*sizeof(float4);

		copy_gldepth_to_depthmap_kernel << <grid, block >> >(gldataptr, depth, s1, s2, camNear);
		cudaSafeCall(cudaGetLastError(), "GpuMesh::copy_gldepth_to_depthmap");
	}


	__global__ void copy_canoview_kernel(PtrStepSz<float4> gldata,
		PtrStepSz<float4> map)
	{
		int u = threadIdx.x + blockIdx.x * blockDim.x;
		int v = threadIdx.y + blockIdx.y * blockDim.y;

		if (u >= map.cols || v >= map.rows)
			return;

		map(v,u) = gldata(map.rows - 1 - v, u);
	}

	void GpuMesh::copy_canoview(const float4* gldata, DeviceArray2D<float4>& map)
	{
		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(map.cols(), block.x);
		grid.y = divUp(map.rows(), block.y);

		PtrStepSz<float4> gldataptr;
		gldataptr.data = (float4*)gldata;
		gldataptr.rows = map.rows();
		gldataptr.cols = map.cols();
		gldataptr.step = map.cols()*sizeof(float4);

		copy_canoview_kernel << <grid, block >> >(gldataptr, map);
		cudaSafeCall(cudaGetLastError(), "GpuMesh::copy_canoview");
	}


	__global__ void copy_warp_node_to_gl_buffer_kernel(
		float4* gldata, int* glindex,
		Tbx::Transfo trans, const float4* nodes, 
		cudaTextureObject_t knnTex, cudaTextureObject_t nodesDqVwTex, 
		float3 origion, float invVoxelSize,
		const KnnIdx* nodesKnn, 
		int n, int node_start_id, bool warp_nodes,
		bool single_level_graph)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;

		if (i < n)
		{
			float4 node = nodes[i*3+2];

			if (warp_nodes)
			{
				Tbx::Dual_quat_cu dq = pack_dual_quat(nodes[i * 3], nodes[i * 3 + 1]);
				Tbx::Vec3 t = trans * dq.transform(Tbx::Point3(node.x, node.y, node.z));
				node.x = t.x;
				node.y = t.y;
				node.z = t.z;
			}

			gldata[i] = node;

			if (glindex && nodesKnn)
			{
				KnnIdx knn = nodesKnn[i];
				int start = 2 * KnnK * i;
				for (int k = 0; k < KnnK; k++)
				{
					int nn = knn_k(knn, k);
					if (nn >= WarpField::MaxNodeNum)
						break;
					glindex[start + k*2 + 0] = i + node_start_id;
					glindex[start + k*2 + 1] = nn + node_start_id + 
						WarpField::MaxNodeNum * (!single_level_graph);
				}
			}
		}
	}

	void GpuMesh::copy_warp_node_to_gl_buffer(float4* gldata, const WarpField* warpField, 
		bool warp_nodes, bool single_level_graph)
	{
		int* glindex = (int*)(gldata + WarpField::MaxNodeNum * WarpField::GraphLevelNum);
		int node_start_id = 0;

		cudaTextureObject_t nodesDqVwTex = warpField->getNodesDqVwTexture();
		cudaTextureObject_t knnTex = warpField->getKnnFieldTexture();
		float3 origion = warpField->getVolume()->getOrigion();
		float invVsz = 1.f/warpField->getVolume()->getVoxelSize();

		for (int lv = 0; lv < warpField->getNumLevels(); lv++, 
			gldata += WarpField::MaxNodeNum, 
			node_start_id += WarpField::MaxNodeNum,
			glindex += WarpField::MaxNodeNum*2*KnnK)
		{
			int n = warpField->getNumNodesInLevel(lv);
			if (n == 0)
				return;
			const float4* nodes = warpField->getNodesDqVwPtr(lv);
			const KnnIdx* indices = nullptr;
			if (lv < warpField->getNumLevels() - 1)
				indices = warpField->getNodesEdgesPtr(lv);
			Tbx::Transfo tr = warpField->get_rigidTransform();
			dim3 block(32);
			dim3 grid(divUp(n, block.x));
			copy_warp_node_to_gl_buffer_kernel << <grid, block >> >(
				gldata, glindex, tr, nodes, 
				knnTex, nodesDqVwTex, origion, invVsz,
				indices, n, node_start_id, warp_nodes, single_level_graph);
			cudaSafeCall(cudaGetLastError(), "GpuMesh::copy_warp_node_to_gl_buffer");
		}
	}

	__global__ void copy_maps_to_gl_buffer_kernel(
		PtrStepSz<float4> vmap_live, PtrStepSz<float4> vmap_warp, 
		PtrStepSz<float4> nmap_live, PtrStepSz<float4> nmap_warp,
		float4* gl_live_v, float4* gl_warp_v, float4* gl_live_n, float4* gl_warp_n,
		int2* gl_edge, float distThre, float angleThreSin, Intr intr)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int h = vmap_warp.rows;
		int w = vmap_warp.cols;

		if (x < vmap_live.cols && y < vmap_live.rows)
		{
			int i = y*vmap_live.cols + x;

			float3 pl = read_float3_4(vmap_live(y, x));
			float3 pw = read_float3_4(vmap_warp(y, x));
			float3 nl = read_float3_4(nmap_live(y, x));
			float3 nw = read_float3_4(nmap_warp(y, x));

			gl_live_v[i] = make_float4(pl.x, pl.y, pl.z, 1.f);
			gl_warp_v[i] = make_float4(pw.x, pw.y, pw.z, 1.f);
			gl_live_n[i] = make_float4(nl.x, nl.y, nl.z, 0.f);
			gl_warp_n[i] = make_float4(nw.x, nw.y, nw.z, 0.f);

			gl_edge[i] = make_int2(0, 0);
			float3 uvd = intr.xyz2uvd(pw);
			int2 ukr = make_int2(uvd.x + 0.5, uvd.y + 0.5);

			// we use opengl coordinate, thus world.z should < 0
			if (ukr.x >= 0 && ukr.y >= 0 && ukr.x < w && ukr.y < h && pw.z < 0)
			{
				float3 plive = read_float3_4(vmap_live[ukr.y*w + ukr.x]);
				float3 nlive = read_float3_4(nmap_live[ukr.y*w + ukr.x]);
				float dist = norm(pw - plive);
				float sine = norm(cross(nw, nlive));
				if (dist <= distThre && sine < angleThreSin)
					gl_edge[i] = make_int2(i + w*h, ukr.y*w + ukr.x);
			}
		}
	}

	void GpuMesh::copy_maps_to_gl_buffer(const MapArr& vmap_live, const MapArr& vmap_warp,
		const MapArr& nmap_live, const MapArr& nmap_warp,
		float4* gldata, const Param& param, Intr intr)
	{
		const int w = vmap_live.cols();
		const int h = vmap_live.rows();

		float4* gl_live_v = gldata;
		float4* gl_warp_v = gldata + w*h;
		float4* gl_live_n = gl_warp_v + w*h;
		float4* gl_warp_n = gl_live_n + w*h;
		int2* gledge = (int2*)(gl_warp_n + w*h);

		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(vmap_live.cols(), block.x);
		grid.y = divUp(vmap_live.rows(), block.y);
		
		copy_maps_to_gl_buffer_kernel << <grid, block >> >(vmap_live, vmap_warp, nmap_live, nmap_warp,
			gl_live_v, gl_warp_v, gl_live_n, gl_warp_n,  gledge, 
			param.fusion_nonRigid_distThre, param.fusion_nonRigid_angleThreSin, intr);
		cudaSafeCall(cudaGetLastError(), "GpuMesh::copy_maps_to_gl_buffer");
	}

	__global__ void update_color_buffer_by_warpField_kernel(
		cudaTextureObject_t knnTex, cudaTextureObject_t nodesDqVwTex,
		GpuMesh::PointType* colors, const GpuMesh::PointType* verts, int num,
		int activeKnnId, float3 origion, float invVoxelSize)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < num)
		{
			float3 p = GpuMesh::from_point(verts[i]);
	
			if (!isnan(p.x))
			{
				GpuMesh::PointType color = GpuMesh::to_point(make_float3(0.7, 0.7, 0.7), 1);
				float3 p1 = (p - origion)*invVoxelSize;
				int x = int(p1.x);
				int y = int(p1.y);
				int z = int(p1.z);
				KnnIdx knnIdx = read_knn_tex(knnTex, x, y, z);
				for (int k = 0; k < KnnK; k++)
				if (knn_k(knnIdx, k) == activeKnnId)
				{
					float4 nearestVw = make_float4(0, 0, 0, 1);
					tex1Dfetch(&nearestVw, nodesDqVwTex, knn_k(knnIdx, k) * 3 + 2); 
					float w = __expf(-norm2(p - GpuMesh::from_point(nearestVw))*0.5f*nearestVw.w*nearestVw.w);
					color = GpuMesh::to_point(make_float3(w, 0, 0));
				}

				colors[i] = color;
			}
		}
	}

	void GpuMesh::update_color_buffer_by_warpField(const WarpField* warpField, GpuMesh* canoMesh)
	{
		if (canoMesh->num() != num())
			throw std::exception("GpuMesh::update_color_buffer_by_warpField(): cano mesh not matched!");
		lockVertsNormals();
		canoMesh->lockVertsNormals();
		cudaTextureObject_t knnTex = warpField->getKnnFieldTexture();
		cudaTextureObject_t nodesDqVwTex = warpField->getNodesDqVwTexture();
		float3 ori = warpField->getVolume()->getOrigion();
		float vsz = warpField->getVolume()->getVoxelSize();
		int aid = warpField->getActiveVisualizeNodeId();

		dim3 block(256);
		dim3 grid(divUp(num(), block.x));
		update_color_buffer_by_warpField_kernel << <grid, block >> >(
			knnTex, nodesDqVwTex, m_colors_d, canoMesh->verts(), num(), aid, ori, 1.f/vsz);
		cudaSafeCall(cudaGetLastError(), "update_color_buffer_by_warpField_kernel");

		canoMesh->unlockVertsNormals();
		unlockVertsNormals();
	}
}