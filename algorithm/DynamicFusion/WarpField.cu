#include "WarpField.h"
#include "GpuMesh.h"
#include "device_utils.h"
namespace dfusion
{
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
}