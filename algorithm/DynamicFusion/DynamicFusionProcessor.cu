#include "DynamicFusionProcessor.h"
#include "TsdfVolume.h"
#include "WarpField.h"
#include "device_utils.h"
namespace dfusion
{
	texture<depthtype, cudaTextureType2D, cudaReadModeElementType> g_depth_tex;
	__global__ void tsdf23(
		const PtrStepSz<depthtype> depth, 
		cudaSurfaceObject_t volumeTex,
		const int3 volume_resolution, 
		const float tranc_dist, 
		float max_weight, 
		const Mat33 Rv2c, 
		const float3 tv2c, 
		const Intr intr,
		const float voxel_size)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int z = threadIdx.z + blockIdx.z * (blockDim.z<<3);

		if (x >= volume_resolution.x || y >= volume_resolution.y)
			return;

#pragma unroll
		for (int block_iter = 0; block_iter < 8; block_iter++, z += blockDim.z)
		{
			if (z >= volume_resolution.z)
				break;

			float3 cxyz = Rv2c*make_float3(x*voxel_size, y*voxel_size, z*voxel_size) + tv2c;
			float3 uvd = intr.xyz2uvd(cxyz);
			int2 coo = make_int2(__float2int_rn(uvd.x), __float2int_rn(uvd.y));

			if (coo.x >= 0 && coo.x < depth.cols && coo.y >= 0 && coo.y < depth.rows)
			{
				float depthVal = tex2D(g_depth_tex, coo.x, coo.y)*0.001f;
				float3 dxyz = intr.uvd2xyz(make_float3(coo.x, coo.y, depthVal));
				float sdf = cxyz.z - dxyz.z;

				if (depthVal > KINECT_NEAREST_METER && sdf >= -tranc_dist)
				{
					float2 tsdf_weight_prev = unpack_tsdf(read_tsdf_surface(volumeTex, x, y, z));

					float tsdf = min(1.0f, sdf / tranc_dist);
					float Wrk = 1;
					float tsdf_new = (tsdf_weight_prev.x * tsdf_weight_prev.y + Wrk * tsdf)
						/ (tsdf_weight_prev.y + Wrk);
					float weight_new = min(tsdf_weight_prev.y + Wrk, max_weight);

					write_tsdf_surface(volumeTex, pack_tsdf(tsdf_new, weight_new), x, y, z);
				}
			}
		}// end for block_iter
	}      // __global__

	void DynamicFusionProcessor::fusion()
	{
		dim3 block(32, 8, 2);
		dim3 grid(divUp(m_volume->getResolution().x, block.x), 
			divUp(m_volume->getResolution().y, block.y),
			divUp(m_volume->getResolution().z, block.z<<3));

		// bind src to texture
		size_t offset;
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<depthtype>();
		cudaBindTexture2D(&offset, &g_depth_tex, m_depth_input.ptr(), &desc, 
			m_depth_input.cols(), m_depth_input.rows(), m_depth_input.step());
		assert(offset == 0);
		cudaSurfaceObject_t surf = m_volume->bindSurface();
		
		Tbx::Transfo tr = m_warpField->get_rigidTransform();
		Tbx::Mat3 Rv2c = tr.get_mat3();
		Tbx::Vec3 tv2c = tr.get_translation() + Rv2c*convert(m_volume->getOrigion());

		tsdf23 << <grid, block >> >(
			m_depth_input, 
			surf, 
			m_volume->getResolution(),
			m_volume->getTsdfTruncDist(),
			m_param.fusion_max_weight,
			convert(Rv2c), 
			convert(tv2c), 
			m_kinect_intr, 
			m_volume->getVoxelSize());

		m_volume->unbindSurface(surf);
		cudaUnbindTexture(&g_depth_tex);

		cudaSafeCall(cudaGetLastError());
	}
}