#include "GpuMesh.h"

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
		cudaSafeCall(cudaGetLastError(), "GpuMesh::copy");
		cudaThreadSynchronize();
	}
}