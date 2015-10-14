#include "GpuGaussNewtonSolver.h"
#include "device_utils.h"
namespace dfusion
{
	texture<WarpField::KnnIdx, cudaTextureType1D, cudaReadModeElementType> g_nodesKnnTex;
	texture<float4, cudaTextureType1D, cudaReadModeElementType> g_nodesVwTex;
	texture<float, cudaTextureType1D, cudaReadModeElementType> g_twistTex;

	__device__ __forceinline__ float4 get_nodesVw(int i)
	{
		return tex1Dfetch(g_nodesVwTex, i);
	}

	__device__ __forceinline__ WarpField::KnnIdx get_nodesKnn(int i)
	{
		return tex1Dfetch(g_nodesKnnTex, i);
	}

	__device__ __forceinline__ void get_twist(int i, Tbx::Vec3& r, Tbx::Vec3& t)
	{
		int i6 = i * 6;
		r.x = tex1Dfetch(g_twistTex, i6++);
		r.y = tex1Dfetch(g_twistTex, i6++);
		r.z = tex1Dfetch(g_twistTex, i6++);
		t.x = tex1Dfetch(g_twistTex, i6++);
		t.y = tex1Dfetch(g_twistTex, i6++);
		t.z = tex1Dfetch(g_twistTex, i6++);
	}

	// map the lower part to full 6x6 matrix
	__constant__ int g_lower_2_full_6x6[21] = {
		0,
		6, 7,
		12, 13, 14,
		18, 19, 20, 21,
		24, 25, 26, 27, 28,
		30, 31, 32, 33, 34, 35
	};
	__constant__ int g_lfull_2_lower_6x6[6][6] = {
		{ 0, -1, -1, -1, -1, -1 },
		{ 1, 2, -1, -1, -1, -1 },
		{ 3, 4, 5, -1, -1, -1 },
		{ 6, 7, 8, 9, -1, -1 },
		{ 10, 11, 12, 13, 14, -1 },
		{ 15, 16, 17, 18, 19, 20 },
	};

#define D_1_DIV_6 0.166666667

	__device__ __forceinline__ float3 read_float3_4(float4 a)
	{
		return make_float3(a.x, a.y, a.z);
	}

	__device__ __forceinline__ float sqr(float a)
	{
		return a*a;
	}

	__device__ __forceinline__ float pow3(float a)
	{
		return a*a*a;
	}

	__device__ __forceinline__ float sign(float a)
	{
		return a>0 - a<0;
	}

	__device__ __forceinline__ WarpField::IdxType& knn_k(WarpField::KnnIdx& knn, int k)
	{
		return ((WarpField::IdxType*)(&knn))[k];
	}
	__device__ __forceinline__ const WarpField::IdxType& knn_k(const WarpField::KnnIdx& knn, int k)
	{
		return ((WarpField::IdxType*)(&knn))[k];
	}

#pragma region --bind textures
	void GpuGaussNewtonSolver::bindTextures()
	{
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<WarpField::KnnIdx>();
			cudaBindTexture(&offset, &g_nodesKnnTex, m_nodesKnn.ptr(), &desc, 
				m_nodesKnn.size() * sizeof(WarpField::KnnIdx));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error!");
		}
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
			cudaBindTexture(&offset, &g_nodesVwTex, m_nodesVw.ptr(), &desc, 
				m_nodesVw.size() * sizeof(float4));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error!");
		}
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cudaBindTexture(&offset, &g_twistTex, m_twist.ptr(), &desc,
				m_twist.size() * sizeof(float));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error!");
		}
	}

	void GpuGaussNewtonSolver::unBindTextures()
	{
		cudaUnbindTexture(g_twistTex);
		cudaUnbindTexture(g_nodesVwTex);
		cudaUnbindTexture(g_nodesKnnTex);
	}
#pragma endregion

#pragma region --calc data term

	struct DataTermCombined
	{
		typedef WarpField::KnnIdx KnnIdx;
		typedef WarpField::IdxType IdxType;
		enum
		{
			CTA_SIZE_X = GpuGaussNewtonSolver::CTA_SIZE_X,
			CTA_SIZE_Y = GpuGaussNewtonSolver::CTA_SIZE_Y,
			CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,
			KnnK = WarpField::KnnK,
			VarPerNode = GpuGaussNewtonSolver::VarPerNode,
			VarPerNode2 = VarPerNode*VarPerNode,
			LowerPartNum = GpuGaussNewtonSolver::LowerPartNum,
		};

		PtrStep<float4> vmap_live;
		PtrStep<float4> nmap_live;
		PtrStep<float4> vmap_warp;
		PtrStep<float4> nmap_warp;
		PtrStep<float4> vmap_cano;
		PtrStep<float4> nmap_cano;
		PtrStep<KnnIdx> vmapKnn;
		float* Hd_;
		float* g_;

		Intr intr;
		Tbx::Transfo Tlw;

		int imgWidth;
		int imgHeight;
		int nNodes;

		float distThres;
		float angleThres;
		float psi_reg;
		float psi_data;

		__device__ __forceinline__ float data_term_energy(float f)const
		{
			return psi_data*psi_data * D_1_DIV_6 * (1 - pow3(max(0.f, 1.f - sqr(f / psi_data))));

			//// the robust Tukey penelty gradient
			//if (abs(f) <= psi_data)
			//	return psi_data*psi_data * D_1_DIV_6 * (1 - pow3(1 - sqr(f / psi_data)));
			//else
			//	return psi_data*psi_data * D_1_DIV_6;
		}

		__device__ __forceinline__ float data_term_penalty(float f)const
		{
			return f * sqr(max(0.f, 1.f - sqr(f / psi_data)));
			//// the robust Tukey penelty gradient
			//if (abs(f) <= psi_data)
			//	return f * sqr(1 - sqr(f / psi_data));
			//else
			//	return 0;
		}

		__device__ __forceinline__ Tbx::Transfo outer_product(Tbx::Vec3 n, Tbx::Point3 v)const
		{
			return Tbx::Transfo(
				n.x*v.x, n.x*v.y, n.x*v.z, n.x,
				n.y*v.x, n.y*v.y, n.y*v.z, n.y,
				n.z*v.x, n.z*v.y, n.z*v.z, n.z,
				0, 0, 0, 0
				);
		}

		__device__ __forceinline__ Tbx::Dual_quat_cu p_qk_p_alpha_func(Tbx::Dual_quat_cu dq, int i)const
		{
			Tbx::Quat_cu q0(0, 0, 0, 0), q1 = dq.get_non_dual_part();

			if (i < 3)
			{
				Tbx::Vec3 t, r;
				float b, c, n;
				dq.to_twist(r, t);
				n = r.norm();
				b = sin(n) / n;
				c = (cos(n) - b) / (n*n);
				if (n > Tbx::Dual_quat_cu::epsilon())
				{
					q0[0] = -r[i] * b;
					q0[1] = r[i] * r[0] * c;
					q0[2] = r[i] * r[1] * c;
					q0[3] = r[i] * r[2] * c;
					q0[i+1] += b;
				}
				else
				{
					q0[i+1] = 1;
				}

				q1.coeff0 = (t.x * q0.coeff1 + t.y * q0.coeff2 + t.z * q0.coeff3) * (-0.5);
				q1.coeff1 = (t.x * q0.coeff0 + t.y * q0.coeff3 - t.z * q0.coeff2) * 0.5;
				q1.coeff2 = (-t.x * q0.coeff3 + t.y * q0.coeff0 + t.z * q0.coeff1) * 0.5;
				q1.coeff3 = (t.x * q0.coeff2 - t.y * q0.coeff1 + t.z * q0.coeff0) * 0.5;
				return Tbx::Dual_quat_cu(q0, q1);
			}
			else
			{
				switch (i)
				{
				case 3:
					return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff1, q1.coeff0, -q1.coeff3, q1.coeff2)*0.5f);
				case 4:
					return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff2, q1.coeff3, q1.coeff0, -q1.coeff1)*0.5f);
				case 5:
					return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff3, -q1.coeff2, q1.coeff1, q1.coeff0)*0.5f);
				default:
					return Tbx::Dual_quat_cu::identity();
				}
			}
		}

		__device__ __forceinline__ float trace_AtB(Tbx::Transfo A, Tbx::Transfo B)const
		{
			float sum = 0;
			for (int i = 0; i < 16; i++)
				sum += A[i] * B[i];
			return sum;
		}

		__device__ __forceinline__ Tbx::Transfo p_SE3_p_dq_func(Tbx::Dual_quat_cu dq, int i)const
		{
			Tbx::Quat_cu q0 = dq.get_non_dual_part();
			Tbx::Quat_cu q1 = dq.get_dual_part();
			float x0 = q0.i(), y0 = q0.j(), z0 = q0.k(), w0 = q0.w();
			float x1 = q1.i(), y1 = q1.j(), z1 = q1.k(), w1 = q1.w();
			switch (i)
			{
			case 0:
				return Tbx::Transfo(
					0, -z0, y0, x1,
					z0, 0, -x0, y1,
					-y0, x0, 0, z1,
					0, 0, 0, 0) * 2;
			case 1:
				return Tbx::Transfo(
					0, y0, z0, -w1,
					y0, -2 * x0, -w0, -z1,
					z0, w0, -2 * x0, y1,
					0, 0, 0, 0) * 2;
			case 2:
				return Tbx::Transfo(
					-2 * y0, x0, w0, z1,
					x0, 0, z0, -w1,
					-w0, z0, -2 * y0, -x1,
					0, 0, 0, 0) * 2;
			case 3:
				return Tbx::Transfo(
					-2 * z0, -w0, x0, -y1,
					w0, -2 * z0, y0, x1,
					x0, y0, 0, -w1,
					0, 0, 0, 0) * 2;
			case 4:
				return Tbx::Transfo(
					0, 0, 0, -x0,
					0, 0, 0, -y0,
					0, 0, 0, -z0,
					0, 0, 0, 0) * 2;
			case 5:
				return Tbx::Transfo(
					0, 0, 0, w0,
					0, 0, 0, z0,
					0, 0, 0, -y0,
					0, 0, 0, 0) * 2;
			case 6:
				return Tbx::Transfo(
					0, 0, 0, -z0,
					0, 0, 0, w0,
					0, 0, 0, x0,
					0, 0, 0, 0) * 2;
			case 7:
				return Tbx::Transfo(
					0, 0, 0, y0,
					0, 0, 0, -x0,
					0, 0, 0, w0,
					0, 0, 0, 0) * 2;
			default:
				printf("index out of range");
				return Tbx::Transfo::identity();
			}
		}

		__device__ __forceinline__ Tbx::Transfo p_SE3_p_alpha_func(Tbx::Dual_quat_cu dq, int i)const
		{
			Tbx::Transfo T = Tbx::Transfo::empty();
			Tbx::Dual_quat_cu p_dq_p_alphai = p_qk_p_alpha_func(dq, i);
			for (int j = 0; j < 8; j++)
				T = T + p_SE3_p_dq_func(dq, j)*p_dq_p_alphai[j];
			return T;
		}

		__device__ __forceinline__ Tbx::Transfo compute_p_f_p_T(const Tbx::Vec3& n,
			const Tbx::Point3& v, const Tbx::Point3& vl, const Tbx::Dual_quat_cu& dq)const
		{
			//Tbx::Transfo T = Tlw*dq.to_transformation_after_normalize();
			//Tbx::Transfo nvt = outer_product(n, v);
			//Tbx::Transfo vlnt = outer_product(n, vl).transpose();
			//Tbx::Transfo p_f_p_T = T*(nvt + nvt.transpose()) - vlnt;
			Tbx::Vec3 Tn = Tlw*dq.rotate(n);
			Tbx::Point3 Tv(Tlw*dq.transform(v) - vl);
			return Tbx::Transfo(
				Tn.x*v.x + n.x*Tv.x, Tn.x*v.y + n.y*Tv.x, Tn.x*v.z + n.z*Tv.x, Tn.x,
				Tn.y*v.x + n.x*Tv.y, Tn.y*v.y + n.y*Tv.y, Tn.y*v.z + n.z*Tv.y, Tn.y,
				Tn.z*v.x + n.x*Tv.z, Tn.z*v.y + n.y*Tv.z, Tn.z*v.z + n.z*Tv.z, Tn.z,
				n.x, n.y, n.z, 0
				);
		}

		__device__ __forceinline__ bool search(int x, int y, Tbx::Point3& vl) const
		{
			float3 vwarp = read_float3_4(vmap_warp(y, x));
			float3 nwarp = read_float3_4(nmap_warp(y, x));

			if (isnan(nwarp.x))
				return false;

			float3 uvd = intr.xyz2uvd(vwarp);
			int2 ukr = make_int2(uvd.x + 0.5, uvd.y + 0.5);

			// we use opengl coordinate, thus world.z should < 0
			if (ukr.x < 0 || ukr.y < 0 || ukr.x >= imgWidth || ukr.y >= imgHeight || vwarp.z >= 0)
				return false;

			float3 vlive = read_float3_4(vmap_live[ukr.y*imgWidth + ukr.x]);
			float3 nlive = read_float3_4(nmap_live[ukr.y*imgWidth + ukr.x]);
			if (isnan(nlive.x))
				return false;

			float dist = norm(vwarp - vlive);
			if (!(dist <= distThres))
				return false;

			float sine = norm(cross(nwarp, nlive));
			if (!(sine <= angleThres))
				return false;

			vl = Tbx::Point3(vlive.x, vlive.y, vlive.z);

			return true;
		}

		__device__ __forceinline__ void operator () () const
		{
			const int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
			const int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

			Tbx::Point3 vl;
			bool found_coresp = false;
			if (x < imgWidth && y < imgHeight)
				found_coresp = search(x, y, vl);

			if (found_coresp)
			{
				Tbx::Point3 v(convert(read_float3_4(vmap_cano(y, x))));
				Tbx::Vec3 n(convert(read_float3_4(nmap_cano(y, x))));

				const KnnIdx knn = vmapKnn(y, x);
				Tbx::Dual_quat_cu dq(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
				Tbx::Dual_quat_cu dqk[KnnK];
				float wk[KnnK];
				for (int k = 0; k < KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						Tbx::Vec3 r, t;
						get_twist(knnNodeId, r, t);
						float4 nodeVw = get_nodesVw(knnNodeId);
						Tbx::Point3 nodesV(convert(read_float3_4(nodeVw)));
						float invNodesW = nodeVw.w;
						dqk[k].from_twist(r, t);
						// note: we store inv radius as vw.w, thus using * instead of / here
						wk[k] = __expf(-(v - nodesV).dot(v - nodesV)*(2 * invNodesW * invNodesW));
						if (dqk[0].get_non_dual_part().dot(dqk[k].get_non_dual_part()) < 0)
							wk[k] = -wk[k];
						dq = dq + dqk[k] * wk[k];
					}
				}

				Tbx::Dual_quat_cu dq_bar = dq;
				float inv_norm_dq_bar = 1.f / dq_bar.get_non_dual_part().norm();
				dq = dq * inv_norm_dq_bar; // normalize

				v = Tlw*dq.transform(v);
				n = Tlw*dq.rotate(n);

				// the grad energy f
				const float f = data_term_penalty(n.dot(v - vl));

				// paitial_f_partial_T
				const Tbx::Transfo p_f_p_T = compute_p_f_p_T(n, v, vl, dq);

				for (int knnK = 0; knnK < KnnK; knnK++)
				{
					float p_f_p_alpha[VarPerNode];
					int knnNodeId = knn_k(knn, knnK);
					float wk_k = wk[knnK] * inv_norm_dq_bar * 2;
					if (knnNodeId < nNodes)
					{
						// partial_T_partial_alphak
						for (int ialpha = 0; ialpha < 6; ialpha++)
						{
							Tbx::Transfo p_T_p_alphak = Tbx::Transfo::empty();
							Tbx::Dual_quat_cu p_qk_p_alpha = p_qk_p_alpha_func(dqk[knnK], ialpha);
							float pdot = dq_bar.get_non_dual_part().dot(p_qk_p_alpha.get_non_dual_part())
								* sqr(inv_norm_dq_bar);
	
							//// evaluate p_dqi_p_alphak, heavily hard code here
							//// this hard code is crucial to the performance 
							// 0:
							// (0, -z0, y0, x1,
							// z0, 0, -x0, y1,
							//-y0, x0, 0, z1,
							// 0, 0, 0, 0) * 2;
							float p_dqi_p_alphak = wk_k * (
								p_qk_p_alpha[0] - dq_bar[0] *  pdot
								);
							p_T_p_alphak[1] += -dq[3] * p_dqi_p_alphak;
							p_T_p_alphak[2] += dq[2] * p_dqi_p_alphak;
							p_T_p_alphak[3] += dq[5] * p_dqi_p_alphak;
							p_T_p_alphak[4] += dq[3] * p_dqi_p_alphak;
							p_T_p_alphak[6] += -dq[1] * p_dqi_p_alphak;
							p_T_p_alphak[7] += dq[6] * p_dqi_p_alphak;
							p_T_p_alphak[8] += -dq[2] * p_dqi_p_alphak;
							p_T_p_alphak[9] += dq[1] * p_dqi_p_alphak;
							p_T_p_alphak[11] += dq[7] * p_dqi_p_alphak;

							// 1
							//( 0, y0, z0, -w1,
							//	y0, -2 * x0, -w0, -z1,
							//	z0, w0, -2 * x0, y1,
							//	0, 0, 0, 0) * 2;
							p_dqi_p_alphak = wk_k * (
								p_qk_p_alpha[1] - dq_bar[1] * pdot
								);
							p_T_p_alphak[1] += dq[2] * p_dqi_p_alphak;
							p_T_p_alphak[2] += dq[3] * p_dqi_p_alphak;
							p_T_p_alphak[3] += -dq[4] * p_dqi_p_alphak;
							p_T_p_alphak[4] += dq[2] * p_dqi_p_alphak;
							p_T_p_alphak[5] += -dq[1] * p_dqi_p_alphak * 2;
							p_T_p_alphak[6] += -dq[0] * p_dqi_p_alphak;
							p_T_p_alphak[7] += -dq[7] * p_dqi_p_alphak;
							p_T_p_alphak[8] += dq[3] * p_dqi_p_alphak;
							p_T_p_alphak[9] += dq[0] * p_dqi_p_alphak;
							p_T_p_alphak[10] += -dq[1] * p_dqi_p_alphak * 2;
							p_T_p_alphak[11] += dq[6] * p_dqi_p_alphak;

							// 2.
							// (-2 * y0, x0, w0, z1,
							//	x0, 0, z0, -w1,
							//	-w0, z0, -2 * y0, -x1,
							//	0, 0, 0, 0) * 2;
							p_dqi_p_alphak = wk_k * (
								p_qk_p_alpha[2] - dq_bar[2] * pdot
								);
							p_T_p_alphak[0] += -dq[2] * p_dqi_p_alphak * 2;
							p_T_p_alphak[1] += dq[1] * p_dqi_p_alphak;
							p_T_p_alphak[2] += dq[0] * p_dqi_p_alphak;
							p_T_p_alphak[3] += dq[7] * p_dqi_p_alphak;
							p_T_p_alphak[4] += dq[1] * p_dqi_p_alphak;
							p_T_p_alphak[6] += dq[3] * p_dqi_p_alphak;
							p_T_p_alphak[7] += -dq[4] * p_dqi_p_alphak;
							p_T_p_alphak[8] += -dq[0] * p_dqi_p_alphak;
							p_T_p_alphak[9] += dq[3] * p_dqi_p_alphak;
							p_T_p_alphak[10] += -dq[2] * p_dqi_p_alphak * 2;
							p_T_p_alphak[11] += -dq[5] * p_dqi_p_alphak;

							// 3.
							// (-2 * z0, -w0, x0, -y1,
							//	w0, -2 * z0, y0, x1,
							//	x0, y0, 0, -w1,
							//	0, 0, 0, 0) * 2;
							p_dqi_p_alphak = wk_k * (
								p_qk_p_alpha[3] - dq_bar[3] * pdot
								);
							p_T_p_alphak[0] += -dq[3] * p_dqi_p_alphak * 2;
							p_T_p_alphak[1] += -dq[0] * p_dqi_p_alphak;
							p_T_p_alphak[2] += dq[1] * p_dqi_p_alphak;
							p_T_p_alphak[3] += -dq[6] * p_dqi_p_alphak;
							p_T_p_alphak[4] += dq[0] * p_dqi_p_alphak;
							p_T_p_alphak[5] += -dq[3] * p_dqi_p_alphak * 2;
							p_T_p_alphak[6] += dq[2] * p_dqi_p_alphak;
							p_T_p_alphak[7] += dq[5] * p_dqi_p_alphak;
							p_T_p_alphak[8] += dq[1] * p_dqi_p_alphak;
							p_T_p_alphak[9] += dq[2] * p_dqi_p_alphak;
							p_T_p_alphak[11] += -dq[4] * p_dqi_p_alphak;

							// 4.
							//( 0, 0, 0, -x0,
							//	0, 0, 0, -y0,
							//	0, 0, 0, -z0,
							//	0, 0, 0, 0) * 2;
							p_dqi_p_alphak = wk_k * (
								p_qk_p_alpha[4] - dq_bar[4] * pdot
								) * 2;
							p_T_p_alphak[3] += -dq[1] * p_dqi_p_alphak;
							p_T_p_alphak[7] += -dq[2] * p_dqi_p_alphak;
							p_T_p_alphak[11] += -dq[3] * p_dqi_p_alphak;

							// 5. 
							// (0, 0, 0, w0,
							//	0, 0, 0, z0,
							//	0, 0, 0, -y0,
							//	0, 0, 0, 0) * 2;
							p_dqi_p_alphak = wk_k * (
								p_qk_p_alpha[5] - dq_bar[5] * pdot
								);
							p_T_p_alphak[3] += dq[0] * p_dqi_p_alphak;
							p_T_p_alphak[7] += dq[3] * p_dqi_p_alphak;
							p_T_p_alphak[11] += -dq[2] * p_dqi_p_alphak;

							// 6. 
							// (0, 0, 0, -z0,
							//	0, 0, 0, w0,
							//	0, 0, 0, x0,
							//	0, 0, 0, 0) * 2;
							p_dqi_p_alphak = wk_k * (
								p_qk_p_alpha[6] - dq_bar[6] * pdot
								);
							p_T_p_alphak[3] += -dq[3] * p_dqi_p_alphak;
							p_T_p_alphak[7] += dq[0] * p_dqi_p_alphak;
							p_T_p_alphak[11] += dq[1] * p_dqi_p_alphak;

							// 7.
							// (0, 0, 0, y0,
							//	0, 0, 0, -x0,
							//	0, 0, 0, w0,
							//	0, 0, 0, 0) * 2;
							p_dqi_p_alphak = wk_k * (
								p_qk_p_alpha[7] - dq_bar[7] * pdot
								);
							p_T_p_alphak[3] += -dq[2] * p_dqi_p_alphak;
							p_T_p_alphak[7] += -dq[1] * p_dqi_p_alphak;
							p_T_p_alphak[11] += dq[0] * p_dqi_p_alphak;

							p_f_p_alpha[ialpha] = trace_AtB(p_f_p_T, Tlw * p_T_p_alphak);
						}// end for ialpha

						// reduce
						int shift = knnNodeId * VarPerNode2;
						int shift_g = knnNodeId * VarPerNode;
						for (int i = 0; i < VarPerNode; ++i)
						{
							#pragma unroll
							for (int j = 0; j <= i; ++j)
								atomicAdd(&Hd_[shift + j], p_f_p_alpha[i] * p_f_p_alpha[j]);
							atomicAdd(&g_[shift_g + i], p_f_p_alpha[i] * f);
							shift += VarPerNode;
						}// end for i
					}// end if knnNodeId < nNodes
				}// end for knnK
			}// end if found corr
		}// end function ()
	};

	__global__ void dataTermCombinedKernel(const DataTermCombined cs)
	{
		cs();
	}

	void GpuGaussNewtonSolver::calcDataTerm()
	{
		DataTermCombined cs;
		cs.angleThres = m_param->fusion_nonRigid_angleThreSin;
		cs.distThres = m_param->fusion_nonRigid_distThre;
		cs.Hd_ = m_Hd;
		cs.g_ = m_g;
		cs.imgHeight = m_vmap_cano->rows();
		cs.imgWidth = m_vmap_cano->cols();
		cs.intr = m_intr;
		cs.nmap_cano = *m_nmap_cano;
		cs.nmap_live = *m_nmap_live;
		cs.nmap_warp = *m_nmap_warp;
		cs.vmap_cano = *m_vmap_cano;
		cs.vmap_live = *m_vmap_live;
		cs.vmap_warp = *m_vmap_warp;
		cs.vmapKnn = m_vmapKnn;
		cs.nNodes = m_numNodes;
		cs.Tlw = m_pWarpField->get_rigidTransform();
		cs.psi_data = m_param->fusion_psi_data;
		cs.psi_reg = m_param->fusion_psi_reg;

		//////////////////////////////
		dim3 block(CTA_SIZE_X, CTA_SIZE_Y);
		dim3 grid(1, 1, 1);
		grid.x = divUp(cs.imgWidth, block.x);
		grid.y = divUp(cs.imgHeight, block.y);
		dataTermCombinedKernel << <grid, block >> >(cs);
		cudaSafeCall(cudaGetLastError(), "dataTermCombinedKernel");
	}
#pragma endregion
}