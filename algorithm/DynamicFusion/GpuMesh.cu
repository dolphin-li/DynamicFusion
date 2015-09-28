#include "GpuMesh.h"
#include "WarpField.h"
namespace dfusion
{
	__device__ __forceinline__ PixelRGBA copy_uchar4_pixelRGBA(uchar4 a)
	{
		PixelRGBA p;
		p.r = a.x;
		p.g = a.y;
		p.b = a.z;
		p.a = a.w;
		return p;
	}

	__global__ void copy_invert_y_kernel(PtrStepSz<uchar4> gldata,
		PtrStepSz<PixelRGBA> img)
	{
		int u = threadIdx.x + blockIdx.x * blockDim.x;
		int v = threadIdx.y + blockIdx.y * blockDim.y;

		if (u >= img.cols || v >= img.rows)
			return;

		img(v, u) = copy_uchar4_pixelRGBA(gldata(img.rows-1-v, u));
	}

	void GpuMesh::copy_invert_y(const uchar4* gldata, ColorMap& img)
	{
		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(m_width, block.x);
		grid.y = divUp(m_height, block.y);

		PtrStepSz<uchar4> gldataptr;
		gldataptr.data = (uchar4*)gldata;
		gldataptr.rows = m_height;
		gldataptr.cols = m_width;
		gldataptr.step = m_width*sizeof(uchar4);

		img.create(m_height, m_width);

		copy_invert_y_kernel << <grid, block >> >(gldataptr, img);
		cudaSafeCall(cudaGetLastError(), "GpuMesh::copy_invert_y");
		cudaThreadSynchronize();
	}

	__global__ void copy_gldepth_to_depthmap_kernel(PtrStepSz<uchar4> gldata,
		PtrStepSz<depthtype> img, float s1, float s2, float camNear)
	{
		int u = threadIdx.x + blockIdx.x * blockDim.x;
		int v = threadIdx.y + blockIdx.y * blockDim.y;

		if (u >= img.cols || v >= img.rows)
			return;

		uchar4 p = gldata(img.rows - 1 - v, u);
		float val = (p.x + (p.y << 8) + p.z / 255.f) / 65525.f;
		val = s1 / (2 * val - 1 + s2) * 1000.f;
		if (val <= camNear*1000.f)
			val = 0;
		img(v, u) = val;
	}

	void GpuMesh::copy_gldepth_to_depthmap(const uchar4* gldata, DepthMap& depth, 
		float s1, float s2, float camNear)
	{
		dim3 block(32, 8);
		dim3 grid(1, 1, 1);
		grid.x = divUp(m_width, block.x);
		grid.y = divUp(m_height, block.y);

		PtrStepSz<uchar4> gldataptr;
		gldataptr.data = (uchar4*)gldata;
		gldataptr.rows = m_height;
		gldataptr.cols = m_width;
		gldataptr.step = m_width*sizeof(uchar4);

		depth.create(m_height, m_width);

		copy_gldepth_to_depthmap_kernel << <grid, block >> >(gldataptr, depth, s1, s2, camNear);
		cudaSafeCall(cudaGetLastError(), "GpuMesh::copy_gldepth_to_depthmap");
		cudaThreadSynchronize();
	}


	__global__ void copy_warp_node_to_gl_buffer_kernel(float4* gldata, 
		Tbx::Transfo trans, const float4* nodes, int n)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;

		if (i < n)
		{
			float4 node = nodes[i];
			Tbx::Vec3 t = trans * Tbx::Point3(node.x, node.y, node.z);
			node.x = t.x;
			node.y = t.y;
			node.z = t.z;
			gldata[i] = node;
		}
	}

	void GpuMesh::copy_warp_node_to_gl_buffer(float4* gldata, const WarpField* warpField)
	{
		int n = warpField->getNumNodesInLevel(0);
		if (n == 0)
			return;
		const float4* nodes = warpField->getNodesVwPtr(0);
		Tbx::Transfo tr = warpField->get_rigidTransform();
		dim3 block(256);
		dim3 grid(1);
		grid.x = divUp(n, block.x);

		copy_warp_node_to_gl_buffer_kernel << <grid, block >> >(gldata, tr, nodes, n);
		cudaSafeCall(cudaGetLastError(), "GpuMesh::copy_warp_node_to_gl_buffer");
	}
}