#include "DynamicFusionProcessor.h"
#include "TsdfVolume.h"
#include "WarpField.h"
#include "device_utils.h"
namespace dfusion
{
	__device__ __forceinline__ int sign(float a)
	{
		return (a > 0) - (a < 0);
	}

	template<int maxK>
	__device__ __forceinline__ static Tbx::Dual_quat_cu calc_dual_quat_blend_on_voxel(
		cudaTextureObject_t knnTex, cudaTextureObject_t nodesDqVwTex,
		int x, int y, int z, float3 origion, float voxelSize, float inv_dw_for_fusion2,
		float nodeRadius, float& fusion_weight)
	{
		Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
		fusion_weight = 0.f;

		// 
		float3 p = make_float3(x*voxelSize, y*voxelSize, z*voxelSize) + origion;
		WarpField::KnnIdx knnIdx = make_ushort4(0, 0, 0, 0);
		tex3D(&knnIdx, knnTex, x, y, z);

		// the first quat
		float4 q0, q1, vw;
		int idk, nn3;
		//Tbx::Dual_quat_cu dq_avg;
		idk = WarpField::get_by_arrayid(knnIdx, 0);
		nn3 = idk * 3;
		tex1Dfetch(&q0, nodesDqVwTex, nn3 + 0);
		tex1Dfetch(&q1, nodesDqVwTex, nn3 + 1);
		tex1Dfetch(&vw, nodesDqVwTex, nn3 + 2);
		float dist2_0 = norm2(make_float3(vw.x - p.x, vw.y - p.y, vw.z - p.z));
		dq_blend = pack_dual_quat(q0, q1);
		fusion_weight += sqrt(dist2_0);

		// the other quats
#pragma unroll
		for (int k = 1; k < maxK; k++)
		{
			idk = WarpField::get_by_arrayid(knnIdx, k);
			
			nn3 = idk * 3;
			tex1Dfetch(&q0, nodesDqVwTex, nn3 + 0);
			tex1Dfetch(&q1, nodesDqVwTex, nn3 + 1);
			tex1Dfetch(&vw, nodesDqVwTex, nn3 + 2);
			Tbx::Dual_quat_cu dq = pack_dual_quat(q0, q1);

			// note: we store 1.f/radius in vw.w
			float dist2 = norm2(make_float3(vw.x - p.x, vw.y - p.y, vw.z - p.z));
			float w = __expf(-(dist2 - dist2_0) * 0.5f * inv_dw_for_fusion2)
				* sign(dq_blend.get_non_dual_part().dot(dq.get_non_dual_part()));
			dq_blend = dq_blend + dq*w;
			fusion_weight += sqrt(dist2);
		}
		dq_blend.normalize();
		fusion_weight = float(maxK) * nodeRadius / fusion_weight;
		return dq_blend;
	}

	template<>
	__device__ __forceinline__ static Tbx::Dual_quat_cu calc_dual_quat_blend_on_voxel<0>(
		cudaTextureObject_t knnTex, cudaTextureObject_t nodesDqVwTex,
		int x, int y, int z, float3 origion, float voxelSize, float inv_dw_for_fusion2,
		float nodeRadius, float& fusion_weight)
	{
		fusion_weight = 1;
		return Tbx::Dual_quat_cu::identity();
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
		float inv_dw_for_fusion2;

		cudaTextureObject_t knnTex;
		cudaTextureObject_t nodesDqVwTex;
		Tbx::Quat_cu Rv2c;
		Tbx::Point3 tv2c;

		template<int maxK>
		__device__ __forceinline__ void fusion(int x, int y, int z)
		{
			float fusion_weight = 0;
			Tbx::Dual_quat_cu dq = calc_dual_quat_blend_on_voxel<maxK>(
				knnTex, nodesDqVwTex, x, y, z, origion, voxel_size, inv_dw_for_fusion2,
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

	template<int maxK>
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

			fs.fusion<maxK>(x, y, z);
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
		fs.inv_dw_for_fusion2 = 1.f / (m_param.warp_param_dw_for_fusion*m_param.warp_param_dw_for_fusion);

		fs.knnTex = m_warpField->bindKnnFieldTexture();
		fs.nodesDqVwTex = m_warpField->bindNodesDqVwTexture();	
		Tbx::Transfo tr = m_warpField->get_rigidTransform();
		fs.Rv2c = tr;
		fs.tv2c = Tbx::Point3(tr.get_translation());

		int maxK = min(WarpField::KnnK, m_warpField->getNumNodesInLevel(0));

		switch (maxK)
		{
		case 0:
			tsdf23<0> << <grid, block >> >(fs);
			break;
		case 1:
			tsdf23<1> << <grid, block >> >(fs);
			break;
		case 2:
			tsdf23<2> << <grid, block >> >(fs);
			break;
		case 3:
			tsdf23<3> << <grid, block >> >(fs);
			break;
		case 4:
			tsdf23<4> << <grid, block >> >(fs);
			break;
		default:
			throw std::exception("non supported KnnK!");
		}

		m_warpField->unBindNodesDqVwTexture(fs.nodesDqVwTex);
		m_warpField->unBindKnnFieldTexture(fs.knnTex);
		m_volume->unbindSurface(fs.volumeTex);
		cudaUnbindTexture(&g_depth_tex);

		cudaSafeCall(cudaGetLastError());
	}
}