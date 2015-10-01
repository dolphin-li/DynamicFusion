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

	__device__ __forceinline__ WarpField::IdxType get_by_arrayid(WarpField::KnnIdx knn, int i)
	{
		return ((WarpField::IdxType*)(&knn))[i];
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
		cudaTextureObject_t knnTex, cudaTextureObject_t nodesDqVwTex, float3 origion, float invVoxelSize,
		const WarpField::KnnIdx* nodesKnn, 
		int n, int node_start_id)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;

		if (i < n)
		{
			float4 node = nodes[i*3+2];

			Tbx::Dual_quat_cu dq_blend = WarpField::calc_dual_quat_blend_on_p(knnTex,
				nodesDqVwTex, make_float3(node.x, node.y, node.z), origion, invVoxelSize);
			Tbx::Vec3 t = trans * dq_blend.transform(Tbx::Point3(node.x, node.y, node.z));
			node.x = t.x;
			node.y = t.y;
			node.z = t.z;

			gldata[i] = node;

			if (glindex && nodesKnn)
			{
				WarpField::IdxType* knnIdx = (WarpField::IdxType*)&(nodesKnn[i]);
				int start = 2 * WarpField::KnnK * i;
				for (int k = 0; k < WarpField::KnnK; k++)
				{
					int nn = knnIdx[k];
					if (nn >= WarpField::MaxNodeNum || nn < 0)
						nn = i-WarpField::MaxNodeNum;
					glindex[start + k*2 + 0] = i + node_start_id;
					glindex[start + k*2 + 1] = nn + node_start_id + WarpField::MaxNodeNum;
				}
			}
		}
	}

	void GpuMesh::copy_warp_node_to_gl_buffer(float4* gldata, const WarpField* warpField)
	{
		int* glindex = (int*)(gldata + WarpField::MaxNodeNum * WarpField::GraphLevelNum);
		int node_start_id = 0;

		cudaTextureObject_t nodesDqVwTex = warpField->bindNodesDqVwTexture();
		cudaTextureObject_t knnTex = warpField->bindKnnFieldTexture();
		float3 origion = warpField->getVolume()->getOrigion();
		float invVsz = 1.f/warpField->getVolume()->getVoxelSize();

		for (int lv = 0; lv < warpField->getNumLevels(); lv++, 
			gldata += WarpField::MaxNodeNum, 
			node_start_id += WarpField::MaxNodeNum,
			glindex += WarpField::MaxNodeNum*2*WarpField::KnnK)
		{
			int n = warpField->getNumNodesInLevel(lv);
			if (n == 0)
				return;
			const float4* nodes = warpField->getNodesDqVwPtr(lv);
			const WarpField::KnnIdx* indices = nullptr;
			if (lv < warpField->getNumLevels() - 1)
				indices = warpField->getNodesEdgesPtr(lv);
			Tbx::Transfo tr = warpField->get_rigidTransform();
			dim3 block(32);
			dim3 grid(divUp(n, block.x));
			copy_warp_node_to_gl_buffer_kernel << <grid, block >> >(
				gldata, glindex, tr, nodes, 
				knnTex, nodesDqVwTex, origion, invVsz,
				indices, n, node_start_id);
			cudaSafeCall(cudaGetLastError(), "GpuMesh::copy_warp_node_to_gl_buffer");
		}
		warpField->unBindKnnFieldTexture(knnTex);
		warpField->unBindNodesDqVwTexture(nodesDqVwTex);
	}

	__global__ void copy_maps_to_gl_buffer_kernel(
		PtrStepSz<float4> vmap_live, PtrStepSz<float4> vmap_warp, 
		PtrStepSz<float4> nmap_live, PtrStepSz<float4> nmap_warp,
		float4* gl_live_v, float4* gl_warp_v, float4* gl_live_n, float4* gl_warp_n,
		int2* gl_edge, float distThre, float angleThreSin)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

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

			float dist = norm(pl - pw);
			float agSin = norm(cross(nl, nw));

			if (dist < distThre && agSin < angleThreSin)
				gl_edge[i] = make_int2(i, i + vmap_live.cols*vmap_live.rows);
			else
				gl_edge[i] = make_int2(0, 0);
		}
	}

	void GpuMesh::copy_maps_to_gl_buffer(const MapArr& vmap_live, const MapArr& vmap_warp,
		const MapArr& nmap_live, const MapArr& nmap_warp,
		float4* gldata, const Param& param)
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
			param.fusion_nonRigid_distThre, param.fusion_nonRigid_angleThreSin);
		cudaSafeCall(cudaGetLastError(), "GpuMesh::copy_maps_to_gl_buffer");
	}
}