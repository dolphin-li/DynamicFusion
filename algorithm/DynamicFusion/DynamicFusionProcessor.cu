#include "DynamicFusionProcessor.h"
#include "TsdfVolume.h"
#include "WarpField.h"
#include "device_utils.h"
namespace dfusion
{
//// could not understand it in sec. 3.2 of the paper; just ignore it currently.
#define ENABLE_ADAPTIVE_FUSION_WEIGHT

	__device__ __forceinline__ static Tbx::Dual_quat_cu calc_dual_quat_blend_on_voxel(
		cudaTextureObject_t knnTex, cudaTextureObject_t nodesDqVwTex,
		int x, int y, int z, float3 origion, float voxelSize,
		float nodeRadius, float& fusion_weight)
	{
		Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
		fusion_weight = 0.f;
		int numK = 0;

		float3 p = make_float3(x*voxelSize, y*voxelSize, z*voxelSize) + origion;
		WarpField::KnnIdx knnIdx = make_ushort4(0, 0, 0, 0);
		tex3D(&knnIdx, knnTex, x, y, z);
		if (knnIdx.x >= WarpField::MaxNodeNum)
		{
			dq_blend = Tbx::Dual_quat_cu::identity();
			fusion_weight = 0.1f;
		}
		else
		{
			Tbx::Dual_quat_cu dq0;
#pragma unroll
			for (int k = 0; k < WarpField::KnnK; k++)
			{
				int idk = WarpField::get_by_arrayid(knnIdx, k);
				if (idk < WarpField::MaxNodeNum)
				{
					WarpField::IdxType nn3 = idk * 3;
					float4 q0, q1, vw;
					tex1Dfetch(&q0, nodesDqVwTex, nn3 + 0);
					tex1Dfetch(&q1, nodesDqVwTex, nn3 + 1);
					tex1Dfetch(&vw, nodesDqVwTex, nn3 + 2);

					// note: we store 1.f/radius in vw.w
					float dist2 = norm2(make_float3(vw.x - p.x, vw.y - p.y, vw.z - p.z));
					float w = __expf(-dist2 * 2 * (vw.w*vw.w));
					Tbx::Dual_quat_cu dq = pack_dual_quat(q0, q1);
					if (k == 0)
						dq0 = dq;
					else
					{
						if (dq0.get_non_dual_part().dot(dq.get_non_dual_part()) < 0)
							w = -w;
					}
					dq_blend = dq_blend + dq*w;
					fusion_weight += sqrt(dist2);
					numK++;
				}
			}
			float norm = dq_blend.get_non_dual_part().norm();
			if (norm < Tbx::Dual_quat_cu::epsilon())
				dq_blend = dq0;
			else
				dq_blend = dq_blend * (1.f/norm);
			fusion_weight /= float(numK) * nodeRadius;
		}

		return dq_blend;
	}


	texture<depthtype, cudaTextureType2D, cudaReadModeElementType> g_depth_tex;
	struct Fusioner
	{
		PtrStepSz<depthtype> depth;

		cudaSurfaceObject_t volumeTex;
		int3 volume_resolution;
		float3 origion;
		float nodeRadius;
		float voxel_size;
		float tranc_dist;
		float max_weight;
		Intr intr;

		cudaTextureObject_t knnTex;
		cudaTextureObject_t nodesDqVwTex;
		Tbx::Quat_cu Rv2c;
		Tbx::Point3 tv2c;

		__device__ __forceinline__ void operator()(int x, int y, int z)
		{
			float fusion_weight = 0;
			Tbx::Dual_quat_cu dq = calc_dual_quat_blend_on_voxel(
				knnTex, nodesDqVwTex, x, y, z, origion, voxel_size, 
				nodeRadius, fusion_weight);

			float3 cxyz = convert(Rv2c.rotate(dq.transform(Tbx::Point3(x*voxel_size + origion.x,
				y*voxel_size+origion.y, z*voxel_size+origion.z))) + tv2c);

			float3 uvd = intr.xyz2uvd(cxyz);
			int2 coo = make_int2(__float2int_rn(uvd.x), __float2int_rn(uvd.y));

			if (uvd.x >= 0 && uvd.x < depth.cols && uvd.y >= 0 && uvd.y < depth.rows)
			{
				float depthVal = tex2D(g_depth_tex, coo.x, coo.y)*0.001f;
				float3 dxyz = intr.uvd2xyz(make_float3(coo.x, coo.y, depthVal));
				float sdf = cxyz.z - dxyz.z;

				if (depthVal > KINECT_NEAREST_METER && sdf >= -tranc_dist)
				{
					float2 tsdf_weight_prev = unpack_tsdf(read_tsdf_surface(volumeTex, x, y, z));

					float tsdf = min(1.0f, sdf / tranc_dist);
					float tsdf_new = (tsdf_weight_prev.x * tsdf_weight_prev.y + fusion_weight * tsdf)
						/ (tsdf_weight_prev.y + fusion_weight);
					float weight_new = min(tsdf_weight_prev.y + fusion_weight, max_weight);

					write_tsdf_surface(volumeTex, pack_tsdf(tsdf_new, weight_new), x, y, z);
				}
			}
		}
	};

	__global__ void tsdf23( Fusioner fs)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int z = threadIdx.z + blockIdx.z * (blockDim.z<<3);

		if (x >= fs.volume_resolution.x || y >= fs.volume_resolution.y)
			return;

#pragma unroll
		for (int block_iter = 0; block_iter < 8; block_iter++, z += blockDim.z)
		{
			if (z >= fs.volume_resolution.z)
				break;

			fs(x, y, z);
		}// end for block_iter
	}      // __global__

	void DynamicFusionProcessor::fusion()
	{
		dim3 block(32, 8, 2);
		dim3 grid(divUp(m_volume->getResolution().x, block.x), 
			divUp(m_volume->getResolution().y, block.y),
			divUp(m_volume->getResolution().z, block.z<<3));

		// bind src to texture
		g_depth_tex.filterMode = cudaFilterModePoint;
		size_t offset;
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<depthtype>();
		cudaBindTexture2D(&offset, &g_depth_tex, m_depth_input.ptr(), &desc, 
			m_depth_input.cols(), m_depth_input.rows(), m_depth_input.step());
		assert(offset == 0);

		Fusioner fs;
		fs.depth = m_depth_input;
		fs.volumeTex = m_volume->bindSurface();
		fs.volume_resolution = m_volume->getResolution();
		fs.origion = m_volume->getOrigion();
		fs.nodeRadius = m_param.warp_radius_search_epsilon;
		fs.voxel_size = m_volume->getVoxelSize();
		fs.tranc_dist = m_volume->getTsdfTruncDist();
		fs.max_weight = m_param.fusion_max_weight;
		fs.intr = m_kinect_intr;

		fs.knnTex = m_warpField->bindKnnFieldTexture();
		fs.nodesDqVwTex = m_warpField->bindNodesDqVwTexture();	
		Tbx::Transfo tr = m_warpField->get_rigidTransform();
		fs.Rv2c = tr;
		fs.tv2c = Tbx::Point3(tr.get_translation());

		tsdf23 << <grid, block >> >(fs);

		m_warpField->unBindNodesDqVwTexture(fs.nodesDqVwTex);
		m_warpField->unBindKnnFieldTexture(fs.knnTex);
		m_volume->unbindSurface(fs.volumeTex);
		cudaUnbindTexture(&g_depth_tex);

		cudaSafeCall(cudaGetLastError());
	}
}