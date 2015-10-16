#include "GpuGaussNewtonSolver.h"
#include "device_utils.h"
#include "cudpp\cudpp_wrapper.h"
#include "cudpp\thrust_wrapper.h"
#include "cudpp\ModerGpuWrapper.h"
namespace dfusion
{
	texture<WarpField::KnnIdx, cudaTextureType1D, cudaReadModeElementType> g_nodesKnnTex;
	texture<float4, cudaTextureType1D, cudaReadModeElementType> g_nodesVwTex;
	texture<float, cudaTextureType1D, cudaReadModeElementType> g_twistTex;
	texture<float, cudaTextureType1D, cudaReadModeElementType> g_JrtValTex;
	texture<int, cudaTextureType1D, cudaReadModeElementType> g_JrtCidxTex;

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

	__device__ __forceinline__ float get_JrtVal(int i)
	{
		return tex1Dfetch(g_JrtValTex, i);
	}
	__device__ __forceinline__ int get_JrtCidx(int i)
	{
		return tex1Dfetch(g_JrtCidxTex, i);
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
	__constant__ int g_lower_2_rowShift_6x6[21] = {
		0,
		1, 1,
		2, 2, 2,
		3, 3, 3, 3,
		4, 4, 4, 4, 4,
		5, 5, 5, 5, 5, 5
	};
	__constant__ int g_lower_2_colShift_6x6[21] = {
		0,
		0, 1,
		0, 1, 2,
		0, 1, 2, 3,
		0, 1, 2, 3, 4,
		0, 1, 2, 3, 4, 5
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
		return (a>0.f) - (a<0.f);
	}

	__device__ __forceinline__ WarpField::IdxType& knn_k(WarpField::KnnIdx& knn, int k)
	{
		return ((WarpField::IdxType*)(&knn))[k];
	}
	__device__ __forceinline__ const WarpField::IdxType& knn_k(const WarpField::KnnIdx& knn, int k)
	{
		return ((WarpField::IdxType*)(&knn))[k];
	}

	__device__ __forceinline__ void sort_knn(WarpField::KnnIdx& knn)
	{
		for (int i = 1; i < WarpField::KnnK; i++)
		{
			WarpField::IdxType x = knn_k(knn,i);
			int	j = i;
			while (j > 0 && knn_k(knn, j - 1) > x)
			{
				knn_k(knn, j) = knn_k(knn, j - 1);
				j = j - 1;
			}
			knn_k(knn, j) = x;
		}
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
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cudaBindTexture(&offset, &g_JrtValTex, m_Jrt_val.ptr(), &desc,
				m_Jrt_val.size() * sizeof(float));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error!");
		}
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
			cudaBindTexture(&offset, &g_JrtCidxTex, m_Jrt_ColIdx.ptr(), &desc,
				m_Jrt_ColIdx.size() * sizeof(int));
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

//#define ENABLE_GPU_DUMP_DEBUG

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
		float psi_data;

#ifdef ENABLE_GPU_DUMP_DEBUG
		// for debug
		float* debug_buffer_pixel_sum2;
		float* debug_buffer_pixel_val;
#endif

		__device__ __forceinline__ float data_term_penalty(float f)const
		{
			return f * sqr(max(0.f, 1.f - sqr(f / psi_data)));
			//// the robust Tukey penelty gradient
			//if (abs(f) <= psi_data)
			//	return f * sqr(1 - sqr(f / psi_data));
			//else
			//	return 0;
		}

		__device__ __forceinline__ float trace_AtB(Tbx::Transfo A, Tbx::Transfo B)const
		{
			float sum = 0;
			for (int i = 0; i < 16; i++)
				sum += A[i] * B[i];
			return sum;
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

		__device__ __forceinline__ Tbx::Transfo p_T_p_alphak_func(const Tbx::Dual_quat_cu& p_qk_p_alpha,
			const Tbx::Dual_quat_cu& dq_bar, const Tbx::Dual_quat_cu& dq, float inv_norm_dq_bar, float wk_k)const
		{
			Tbx::Transfo p_T_p_alphak = Tbx::Transfo::empty();

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
				p_qk_p_alpha[0] - dq_bar[0] * pdot
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
				);
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
			p_T_p_alphak[3] += dq[2] * p_dqi_p_alphak;
			p_T_p_alphak[7] += -dq[1] * p_dqi_p_alphak;
			p_T_p_alphak[11] += dq[0] * p_dqi_p_alphak;

			p_T_p_alphak = Tlw * p_T_p_alphak;
			return p_T_p_alphak;
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
			if (!(sine < angleThres))
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
				Tbx::Dual_quat_cu dqk_0;
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
						Tbx::Dual_quat_cu dqk_k;
						dqk_k.from_twist(r, t);
						// note: we store inv radius as vw.w, thus using * instead of / here
						wk[k] = __expf(-(v - nodesV).dot(v - nodesV)*(2 * invNodesW * invNodesW));
						if (k == 0)
							dqk_0 = dqk_k;
						if (dqk_0.get_non_dual_part().dot(dqk_k.get_non_dual_part()) < 0)
							wk[k] = -wk[k];
						dq = dq + dqk_k * wk[k];
					}
				}

				Tbx::Dual_quat_cu dq_bar = dq;
				float inv_norm_dq_bar = 1.f / dq_bar.get_non_dual_part().norm();
				dq = dq * inv_norm_dq_bar; // normalize

				// the grad energy f
				const float f = data_term_penalty((Tlw*dq.rotate(n)).dot(Tlw*dq.transform(v) - vl));

				// paitial_f_partial_T
				const Tbx::Transfo p_f_p_T = compute_p_f_p_T(n, v, vl, dq);

				for (int knnK = 0; knnK < KnnK; knnK++)
				{
					float p_f_p_alpha[VarPerNode];
					int knnNodeId = knn_k(knn, knnK);
					float wk_k = wk[knnK] * inv_norm_dq_bar * 2;
					if (knnNodeId < nNodes)
					{
						//// comput partial_T_partial_alphak, hard code here.
						Tbx::Dual_quat_cu p_qk_p_alpha;
						Tbx::Transfo p_T_p_alphak;
						Tbx::Vec3 t, r;
						float b, c;
						Tbx::Quat_cu q1;
						get_twist(knnNodeId, r, t);
						{
							float n = r.norm();
							float sin_n, cos_n;
							sincos(n, &sin_n, &cos_n);
							b = n > Tbx::Dual_quat_cu::epsilon() ? sin_n / n : 1;
							c = n > Tbx::Dual_quat_cu::epsilon() ? (cos_n - b) / (n*n) : 0;
							q1 = Tbx::Quat_cu(cos_n*0.5f, r.x*b*0.5f, r.y*b*0.5f, r.z*b*0.5f);
						}

						// alpha0
						p_qk_p_alpha[0] = -r[0] * b;
						p_qk_p_alpha[1] = b + r[0] * r[0] * c;
						p_qk_p_alpha[2] = r[0] * r[1] * c;
						p_qk_p_alpha[3] = r[0] * r[2] * c;
						p_qk_p_alpha = Tbx::Dual_quat_cu::dual_quat_from(p_qk_p_alpha.get_non_dual_part(), t);
						p_T_p_alphak = p_T_p_alphak_func(p_qk_p_alpha, dq_bar, dq,
							inv_norm_dq_bar, wk_k);
						p_f_p_alpha[0] = trace_AtB(p_f_p_T, p_T_p_alphak);

						// alpha1
						p_qk_p_alpha[0] = -r[1] * b;
						p_qk_p_alpha[1] = r[1] * r[0] * c;
						p_qk_p_alpha[2] = b + r[1] * r[1] * c;
						p_qk_p_alpha[3] = r[1] * r[2] * c;
						p_qk_p_alpha = Tbx::Dual_quat_cu::dual_quat_from(p_qk_p_alpha.get_non_dual_part(), t);
						p_T_p_alphak = p_T_p_alphak_func(p_qk_p_alpha, dq_bar, dq,
							inv_norm_dq_bar, wk_k);
						p_f_p_alpha[1] = trace_AtB(p_f_p_T, p_T_p_alphak);

						// alpha2
						p_qk_p_alpha[0] = -r[2] * b;
						p_qk_p_alpha[1] = r[2] * r[0] * c;
						p_qk_p_alpha[2] = r[2] * r[1] * c;
						p_qk_p_alpha[3] = b + r[2] * r[2] * c;
						p_qk_p_alpha = Tbx::Dual_quat_cu::dual_quat_from(p_qk_p_alpha.get_non_dual_part(), t);
						p_T_p_alphak = p_T_p_alphak_func(p_qk_p_alpha, dq_bar, dq,
							inv_norm_dq_bar, wk_k);
						p_f_p_alpha[2] = trace_AtB(p_f_p_T, p_T_p_alphak);

						// alpha3
						p_qk_p_alpha = Tbx::Dual_quat_cu(Tbx::Quat_cu(0, 0, 0, 0),
							Tbx::Quat_cu(-q1[1], q1[0], -q1[3], q1[2]));
						p_T_p_alphak = p_T_p_alphak_func(p_qk_p_alpha, dq_bar, dq,
							inv_norm_dq_bar, wk_k);
						p_f_p_alpha[3] = trace_AtB(p_f_p_T, p_T_p_alphak);

						// alpha4
						p_qk_p_alpha = Tbx::Dual_quat_cu(Tbx::Quat_cu(0, 0, 0, 0),
							Tbx::Quat_cu(-q1[2], q1[3], q1[0], -q1[1]));
						p_T_p_alphak = p_T_p_alphak_func(p_qk_p_alpha, dq_bar, dq,
							inv_norm_dq_bar, wk_k);
						p_f_p_alpha[4] = trace_AtB(p_f_p_T, p_T_p_alphak);

						// alpha5
						p_qk_p_alpha = Tbx::Dual_quat_cu(Tbx::Quat_cu(0, 0, 0, 0),
							Tbx::Quat_cu(-q1[3], -q1[2], q1[1], q1[0]));
						p_T_p_alphak = p_T_p_alphak_func(p_qk_p_alpha, dq_bar, dq,
							inv_norm_dq_bar, wk_k);
						p_f_p_alpha[5] = trace_AtB(p_f_p_T, p_T_p_alphak);

						//// reduce--------------------------------------------------
						int shift = knnNodeId * VarPerNode2;
						int shift_g = knnNodeId * VarPerNode;
						for (int i = 0; i < VarPerNode; ++i)
						{
#pragma unroll
							for (int j = 0; j <= i; ++j)
							{
								atomicAdd(&Hd_[shift + j], p_f_p_alpha[i] * p_f_p_alpha[j]);
#ifdef ENABLE_GPU_DUMP_DEBUG
// debug
if (knnNodeId == 390 && i == 5 && j == 1
	&& debug_buffer_pixel_sum2 && debug_buffer_pixel_val
	)
{
	for (int k = 0; k < VarPerNode; k++)
		debug_buffer_pixel_val[(y*imgWidth + x)*VarPerNode + k] =
		p_f_p_alpha[k];
	debug_buffer_pixel_sum2[y*imgWidth + x] = Hd_[shift + j];
}
#endif
							}
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

#ifdef ENABLE_GPU_DUMP_DEBUG
		// debugging
		DeviceArray<float> pixelSum2, pixelVal;
		pixelSum2.create(cs.imgHeight*cs.imgWidth);
		cudaMemset(pixelSum2.ptr(), 0, pixelSum2.sizeBytes());
		pixelVal.create(cs.imgHeight*cs.imgWidth*VarPerNode);
		cudaMemset(pixelVal.ptr(), 0, pixelVal.sizeBytes());
		cs.debug_buffer_pixel_sum2 = pixelSum2;
		cs.debug_buffer_pixel_val = pixelVal;
#endif

		//////////////////////////////
		dim3 block(CTA_SIZE_X, CTA_SIZE_Y);
		dim3 grid(1, 1, 1);
		grid.x = divUp(cs.imgWidth, block.x);
		grid.y = divUp(cs.imgHeight, block.y);
		dataTermCombinedKernel << <grid, block >> >(cs);
		cudaSafeCall(cudaGetLastError(), "dataTermCombinedKernel");

		// debugging
#ifdef ENABLE_GPU_DUMP_DEBUG
		{
			std::vector<float> ps, pv;
			pixelSum2.download(ps);
			pixelVal.download(pv);

			FILE* pFile = fopen("D:/tmp/gpu_pixel.txt", "w");
			for (int i = 0; i < ps.size(); i++)
			{
				fprintf(pFile, "%ef %ef %ef %ef %ef %ef %ef\n",
					pv[i * 6 + 0], pv[i * 6 + 1], pv[i * 6 + 2],
					pv[i * 6 + 3], pv[i * 6 + 4], pv[i * 6 + 5],
					ps[i]);
			}
			fclose(pFile);
		}
#endif
	}
#pragma endregion

#pragma region --define sparse structure
	__global__ void count_Jr_rows_kernel(int* rctptr, int nMaxNodes)
	{
		int i = threadIdx.x + blockIdx.x*blockDim.x;
		if (i < nMaxNodes)
		{
			WarpField::KnnIdx knn = get_nodesKnn(i);
			int numK = -1;
			for (int k = 0; k < WarpField::KnnK; ++k)
			{
				if (knn_k(knn, k) < nMaxNodes)
					numK = k;
			}

			// each node generate 6*maxK rows
			rctptr[i] = (numK+1) * 6;
		}
		if (i == nMaxNodes)
			rctptr[i] = 0;
	}

	__global__ void compute_row_map_kernel(GpuGaussNewtonSolver::JrRow2NodeMapper* row2nodeId, 
		const int* rctptr, int nMaxNodes)
	{
		int iNode = threadIdx.x + blockIdx.x*blockDim.x;
		if (iNode < nMaxNodes)
		{
			int row_b = rctptr[iNode];
			int row_e = rctptr[iNode+1];
			for (int r = row_b; r < row_e; r++)
			{
				GpuGaussNewtonSolver::JrRow2NodeMapper mp;
				mp.nodeId = iNode;
				mp.k = (r - row_b) / 6;
				mp.ixyz = r - 6 * mp.k;
				row2nodeId[r] = mp;
			}
		}
	}

	__global__ void compute_Jr_rowPtr_colIdx_kernel(
		int* rptr, int* rptr_coo, int* colIdx,
		const GpuGaussNewtonSolver::JrRow2NodeMapper* row2nodeId, 
		int nMaxNodes, int nRows)
	{
		enum{
			VarPerNode = GpuGaussNewtonSolver::VarPerNode,
			ColPerRow = VarPerNode * 2
		};
		const int iRow = threadIdx.x + blockIdx.x*blockDim.x;
		if (iRow < nRows)
		{
			const int iNode = row2nodeId[iRow].nodeId;
			if (iNode < nMaxNodes)
			{
				WarpField::KnnIdx knn = get_nodesKnn(iNode);
				int knnNodeId = knn_k(knn, row2nodeId[iRow].k);
				if (knnNodeId < nMaxNodes)
				{
					int col_b = iRow*ColPerRow;
					rptr[iRow] = col_b;

					// each row 2*VerPerNode Cols
					// 1. self
					for (int j = 0; j < VarPerNode; j++, col_b++)
					{
						rptr_coo[col_b] = iRow;
						colIdx[col_b] = iNode*VarPerNode + j;
					}// j
					// 2. neighbor
					for (int j = 0; j < VarPerNode; j++, col_b++)
					{
						rptr_coo[col_b] = iRow;
						colIdx[col_b] = knnNodeId*VarPerNode + j;
					}// j
				}// end if knnNodeId
			}
		}// end if iRow < nRows
		if (iRow == nRows)
			rptr[nRows] = nRows * ColPerRow;
	}

	__global__ void calc_B_cidx_kernel(int* B_rptr_coo, int* B_cidx, 
		const int* B_rptr, int nRows, int nMaxNodes, int nLv0Nodes)
	{
		int iRow = threadIdx.x + blockIdx.x*blockDim.x;
		if (iRow < nRows)
		{
			int iNode = iRow / GpuGaussNewtonSolver::VarPerNode;

			WarpField::KnnIdx knn = get_nodesKnn(iNode);
			int col_b = B_rptr[iRow];
			for (int k = 0; k < WarpField::KnnK; ++k)
			{
				int knnNodeId = knn_k(knn, k);
				if (knnNodeId < nMaxNodes)
				{
					// 2. neighbor
					for (int j = 0; j < GpuGaussNewtonSolver::VarPerNode; j++, col_b++)
					{
						B_rptr_coo[col_b] = iRow;
						B_cidx[col_b] = (knnNodeId-nLv0Nodes)*GpuGaussNewtonSolver::VarPerNode + j;
					}// j
				}
			}
		}
	}

	void GpuGaussNewtonSolver::initSparseStructure()
	{
		// 1. compute Jr structure ==============================================
		// 1.0. decide the total rows we have for each nodes
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_numNodes, block.x));
			count_Jr_rows_kernel << <grid, block >> >(m_Jr_RowCounter.ptr(), m_numNodes);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::initSparseStructure::count_Jr_rows_kernel");
			thrust_wrapper::exclusive_scan(m_Jr_RowCounter.ptr(), m_Jr_RowCounter.ptr(), m_numNodes + 1);
			cudaSafeCall(cudaMemcpy(&m_Jrrows, m_Jr_RowCounter.ptr() + m_numNodes,
				sizeof(int), cudaMemcpyDeviceToHost), "copy Jr rows to host");
		}

		// 1.1. collect nodes edges info:
		//	each low-level nodes are connected to k higher level nodes
		//	but the connections are not stored for the higher level nodes
		//  thus when processing each node, we add 2*k edges, w.r.t. 2*k*3 rows: each (x,y,z) a row
		//	for each row, there are exactly 2*VarPerNode values
		//	after this step, we can get the CSR/COO structure
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_numNodes, block.x));
			compute_row_map_kernel << <grid, block >> >(m_Jr_RowMap2NodeId.ptr(), m_Jr_RowCounter.ptr(), m_numNodes);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::initSparseStructure::compute_row_map_kernel");
		}
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_Jrrows, block.x));
			compute_Jr_rowPtr_colIdx_kernel << <grid, block >> >(m_Jr_RowPtr.ptr(),
				m_Jr_RowPtr_coo.ptr(), m_Jr_ColIdx.ptr(), m_Jr_RowMap2NodeId.ptr(), m_numNodes, m_Jrrows);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::initSparseStructure::compute_Jr_rowPtr_kernel");
			cudaSafeCall(cudaMemcpy(&m_Jrnnzs, m_Jr_RowPtr.ptr() + m_Jrrows,
				sizeof(int), cudaMemcpyDeviceToHost), "copy Jr nnz to host");
		}

		// 2. compute Jrt structure ==============================================
		// 2.1. fill (row, col) as (col, row) from Jr and sort.
		cudaMemcpy(m_Jrt_RowPtr_coo.ptr(), m_Jr_ColIdx.ptr(), m_Jrnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(m_Jrt_ColIdx.ptr(), m_Jr_RowPtr_coo.ptr(), m_Jrnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
		// !!!NOTE: we must use mergesort here, it can guarentees the order of values of the same key
		modergpu_wrapper::mergesort_by_key(m_Jrt_RowPtr_coo.ptr(), m_Jrt_ColIdx.ptr(), m_Jrnnzs);
		cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::initSparseStructure::mergesort_by_key");

		// 2.2. extract CSR rowptr info.
		if (CUSPARSE_STATUS_SUCCESS != cusparseXcoo2csr(m_cuSparseHandle,
			m_Jrt_RowPtr_coo.ptr(), m_Jrnnzs, m_Jrcols,
			m_Jrt_RowPtr.ptr(), CUSPARSE_INDEX_BASE_ZERO))
			throw std::exception("GpuGaussNewtonSolver::initSparseStructure::cusparseXcoo2csr failed");

		// 3. compute B structure ==============================================
		// 3.1 the row ptr of B is the same with the first L0 rows of Jrt.
		cudaMemcpy(m_B_RowPtr.ptr(), m_Jrt_RowPtr.ptr(), (m_Brows + 1)*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaSafeCall(cudaMemcpy(&m_Bnnzs, m_B_RowPtr.ptr() + m_Brows,
			sizeof(int), cudaMemcpyDeviceToHost), "copy B nnz to host");
		
		// 3.2 the col-idx of B
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_Brows, block.x));
			calc_B_cidx_kernel << <grid, block >> >(m_B_RowPtr_coo.ptr(),
				m_B_ColIdx.ptr(), m_B_RowPtr.ptr(), m_Brows, m_numNodes, m_numLv0Nodes);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::initSparseStructure::calc_B_cidx_kernel");
		}

		// 3.3 sort to compute Bt
		cudaMemcpy(m_Bt_RowPtr_coo.ptr(), m_B_ColIdx.ptr(), m_Bnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(m_Bt_ColIdx.ptr(), m_B_RowPtr_coo.ptr(), m_Bnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
		modergpu_wrapper::mergesort_by_key(m_Bt_RowPtr_coo.ptr(), m_Bt_ColIdx.ptr(), m_Bnnzs);
		cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::initSparseStructure::mergesort_by_key");
		if (CUSPARSE_STATUS_SUCCESS != cusparseXcoo2csr(m_cuSparseHandle,
			m_Bt_RowPtr_coo.ptr(), m_Bnnzs, m_Bcols,
			m_Bt_RowPtr.ptr(), CUSPARSE_INDEX_BASE_ZERO))
			throw std::exception("GpuGaussNewtonSolver::initSparseStructure::cusparseXcoo2csr failed");
	}

#pragma endregion

#pragma region --calc reg term
	struct RegTermJacobi
	{
		typedef WarpField::KnnIdx KnnIdx;
		typedef WarpField::IdxType IdxType;
		typedef GpuGaussNewtonSolver::JrRow2NodeMapper Mapper;
		enum
		{
			KnnK = WarpField::KnnK,
			VarPerNode = GpuGaussNewtonSolver::VarPerNode,
			VarPerNode2 = VarPerNode*VarPerNode,
			ColPerRow = VarPerNode * 2
		};

		int nNodes;
		int nRows;
		const Mapper* rows2nodeIds;
		const int* rptr;
		const int* cidx;
		mutable float* vptr;
		mutable float* fptr;

		float psi_reg;
		float lambda;

		__device__ __forceinline__  Tbx::Dual_quat_cu p_qk_p_alpha_func(Tbx::Dual_quat_cu dq, int i)const
		{
			Tbx::Vec3 t, r;
			float b, c, n;
			Tbx::Quat_cu q0(0, 0, 0, 0), q1 = dq.get_non_dual_part();
			switch (i)
			{
			case 0:
				dq.to_twist(r, t);
				n = r.norm();
				if (n > Tbx::Dual_quat_cu::epsilon())
				{
					b = sin(n) / n;
					c = (cos(n) - b) / (n*n);
					q0.coeff0 = -r.x * b;
					q0.coeff1 = b + r.x*r.x*c;
					q0.coeff2 = r.x*r.y*c;
					q0.coeff3 = r.x*r.z*c;
				}
				else
				{
					q0.coeff0 = 0;
					q0.coeff1 = 1;
					q0.coeff2 = 0;
					q0.coeff3 = 0;
				}

				q1.coeff0 = (t.x * q0.coeff1 + t.y * q0.coeff2 + t.z * q0.coeff3) * (-0.5);
				q1.coeff1 = (t.x * q0.coeff0 + t.y * q0.coeff3 - t.z * q0.coeff2) * 0.5;
				q1.coeff2 = (-t.x * q0.coeff3 + t.y * q0.coeff0 + t.z * q0.coeff1) * 0.5;
				q1.coeff3 = (t.x * q0.coeff2 - t.y * q0.coeff1 + t.z * q0.coeff0) * 0.5;
				return Tbx::Dual_quat_cu(q0, q1);
			case 1:
				dq.to_twist(r, t);
				n = r.norm();
				if (n > Tbx::Dual_quat_cu::epsilon())
				{
					b = sin(n) / n;
					c = (cos(n) - b) / (n*n);
					q0.coeff0 = -r.y * b;
					q0.coeff1 = r.y*r.x*c;
					q0.coeff2 = b + r.y*r.y*c;
					q0.coeff3 = r.y*r.z*c;
				}
				else
				{
					q0.coeff0 = 0;
					q0.coeff1 = 0;
					q0.coeff2 = 1;
					q0.coeff3 = 0;
				}

				q1.coeff0 = (t.x * q0.coeff1 + t.y * q0.coeff2 + t.z * q0.coeff3) * (-0.5);
				q1.coeff1 = (t.x * q0.coeff0 + t.y * q0.coeff3 - t.z * q0.coeff2) * 0.5;
				q1.coeff2 = (-t.x * q0.coeff3 + t.y * q0.coeff0 + t.z * q0.coeff1) * 0.5;
				q1.coeff3 = (t.x * q0.coeff2 - t.y * q0.coeff1 + t.z * q0.coeff0) * 0.5;
				return Tbx::Dual_quat_cu(q0, q1);
			case 2:
				dq.to_twist(r, t);
				n = r.norm();
				if (n > Tbx::Dual_quat_cu::epsilon())
				{
					b = sin(n) / n;
					c = (cos(n) - b) / (n*n);

					q0.coeff0 = -r.z * b;
					q0.coeff1 = r.z*r.x*c;
					q0.coeff2 = r.z*r.y*c;
					q0.coeff3 = b + r.z*r.z*c;
				}
				else
				{
					q0.coeff0 = 0;
					q0.coeff1 = 0;
					q0.coeff2 = 0;
					q0.coeff3 = 1;
				}

				q1.coeff0 = (t.x * q0.coeff1 + t.y * q0.coeff2 + t.z * q0.coeff3) * (-0.5);
				q1.coeff1 = (t.x * q0.coeff0 + t.y * q0.coeff3 - t.z * q0.coeff2) * 0.5;
				q1.coeff2 = (-t.x * q0.coeff3 + t.y * q0.coeff0 + t.z * q0.coeff1) * 0.5;
				q1.coeff3 = (t.x * q0.coeff2 - t.y * q0.coeff1 + t.z * q0.coeff0) * 0.5;
				return Tbx::Dual_quat_cu(q0, q1);
			case 3:
				return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff1, q1.coeff0, -q1.coeff3, q1.coeff2))*0.5;
			case 4:
				return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff2, q1.coeff3, q1.coeff0, -q1.coeff1))*0.5;
			case 5:
				return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff3, -q1.coeff2, q1.coeff1, q1.coeff0))*0.5;
			default:
				return Tbx::Dual_quat_cu();
			}
		}

		__device__ __forceinline__  Tbx::Vec3 reg_term_penalty(Tbx::Vec3 f)const
		{
			// the robust Huber penelty gradient
			Tbx::Vec3 df;
			float norm = f.norm();
			if (norm < psi_reg)
				df = f;
			else
			for (int k = 0; k < 3; k++)
				df[k] = sign(f[k])*psi_reg;
			return df;
		}

		__device__ __forceinline__  Tbx::Transfo p_SE3_p_alpha_func(Tbx::Dual_quat_cu dq, int i)const
		{
			Tbx::Transfo T = Tbx::Transfo::empty();
			Tbx::Dual_quat_cu p_dq_p_alphai = p_qk_p_alpha_func(dq, i) * 2.f;

			//// evaluate p_dqi_p_alphak, heavily hard code here
			//// this hard code is crucial to the performance 
			// 0:
			// (0, -z0, y0, x1,
			// z0, 0, -x0, y1,
			//-y0, x0, 0, z1,
			// 0, 0, 0, 0) * 2;
			float p_dqi_p_alphak = p_dq_p_alphai[0];
			T[1] += -dq[3] * p_dqi_p_alphak;
			T[2] += dq[2] * p_dqi_p_alphak;
			T[3] += dq[5] * p_dqi_p_alphak;
			T[4] += dq[3] * p_dqi_p_alphak;
			T[6] += -dq[1] * p_dqi_p_alphak;
			T[7] += dq[6] * p_dqi_p_alphak;
			T[8] += -dq[2] * p_dqi_p_alphak;
			T[9] += dq[1] * p_dqi_p_alphak;
			T[11] += dq[7] * p_dqi_p_alphak;

			// 1
			//( 0, y0, z0, -w1,
			//	y0, -2 * x0, -w0, -z1,
			//	z0, w0, -2 * x0, y1,
			//	0, 0, 0, 0) * 2;
			p_dqi_p_alphak = p_dq_p_alphai[1];
			T[1] += dq[2] * p_dqi_p_alphak;
			T[2] += dq[3] * p_dqi_p_alphak;
			T[3] += -dq[4] * p_dqi_p_alphak;
			T[4] += dq[2] * p_dqi_p_alphak;
			T[5] += -dq[1] * p_dqi_p_alphak * 2;
			T[6] += -dq[0] * p_dqi_p_alphak;
			T[7] += -dq[7] * p_dqi_p_alphak;
			T[8] += dq[3] * p_dqi_p_alphak;
			T[9] += dq[0] * p_dqi_p_alphak;
			T[10] += -dq[1] * p_dqi_p_alphak * 2;
			T[11] += dq[6] * p_dqi_p_alphak;

			// 2.
			// (-2 * y0, x0, w0, z1,
			//	x0, 0, z0, -w1,
			//	-w0, z0, -2 * y0, -x1,
			//	0, 0, 0, 0) * 2;
			p_dqi_p_alphak = p_dq_p_alphai[2];
			T[0] += -dq[2] * p_dqi_p_alphak * 2;
			T[1] += dq[1] * p_dqi_p_alphak;
			T[2] += dq[0] * p_dqi_p_alphak;
			T[3] += dq[7] * p_dqi_p_alphak;
			T[4] += dq[1] * p_dqi_p_alphak;
			T[6] += dq[3] * p_dqi_p_alphak;
			T[7] += -dq[4] * p_dqi_p_alphak;
			T[8] += -dq[0] * p_dqi_p_alphak;
			T[9] += dq[3] * p_dqi_p_alphak;
			T[10] += -dq[2] * p_dqi_p_alphak * 2;
			T[11] += -dq[5] * p_dqi_p_alphak;

			// 3.
			// (-2 * z0, -w0, x0, -y1,
			//	w0, -2 * z0, y0, x1,
			//	x0, y0, 0, -w1,
			//	0, 0, 0, 0) * 2;
			p_dqi_p_alphak = p_dq_p_alphai[3];
			T[0] += -dq[3] * p_dqi_p_alphak * 2;
			T[1] += -dq[0] * p_dqi_p_alphak;
			T[2] += dq[1] * p_dqi_p_alphak;
			T[3] += -dq[6] * p_dqi_p_alphak;
			T[4] += dq[0] * p_dqi_p_alphak;
			T[5] += -dq[3] * p_dqi_p_alphak * 2;
			T[6] += dq[2] * p_dqi_p_alphak;
			T[7] += dq[5] * p_dqi_p_alphak;
			T[8] += dq[1] * p_dqi_p_alphak;
			T[9] += dq[2] * p_dqi_p_alphak;
			T[11] += -dq[4] * p_dqi_p_alphak;

			// 4.
			//( 0, 0, 0, -x0,
			//	0, 0, 0, -y0,
			//	0, 0, 0, -z0,
			//	0, 0, 0, 0) * 2;
			p_dqi_p_alphak = p_dq_p_alphai[4];
			T[3] += -dq[1] * p_dqi_p_alphak;
			T[7] += -dq[2] * p_dqi_p_alphak;
			T[11] += -dq[3] * p_dqi_p_alphak;

			// 5. 
			// (0, 0, 0, w0,
			//	0, 0, 0, z0,
			//	0, 0, 0, -y0,
			//	0, 0, 0, 0) * 2;
			p_dqi_p_alphak = p_dq_p_alphai[5];
			T[3] += dq[0] * p_dqi_p_alphak;
			T[7] += dq[3] * p_dqi_p_alphak;
			T[11] += -dq[2] * p_dqi_p_alphak;

			// 6. 
			// (0, 0, 0, -z0,
			//	0, 0, 0, w0,
			//	0, 0, 0, x0,
			//	0, 0, 0, 0) * 2;
			p_dqi_p_alphak = p_dq_p_alphai[6];
			T[3] += -dq[3] * p_dqi_p_alphak;
			T[7] += dq[0] * p_dqi_p_alphak;
			T[11] += dq[1] * p_dqi_p_alphak;

			// 7.
			// (0, 0, 0, y0,
			//	0, 0, 0, -x0,
			//	0, 0, 0, w0,
			//	0, 0, 0, 0) * 2;
			p_dqi_p_alphak = p_dq_p_alphai[7];
			T[3] += dq[2] * p_dqi_p_alphak;
			T[7] += -dq[1] * p_dqi_p_alphak;
			T[11] += dq[0] * p_dqi_p_alphak;

			return T;
		}

		__device__ __forceinline__ void operator () () const
		{
			const int iRow = (threadIdx.x + blockIdx.x * blockDim.x)*6;
		
			if (iRow >= nRows)
				return;

			Mapper mapper = rows2nodeIds[iRow];
			int knnNodeId = knn_k(get_nodesKnn(mapper.nodeId), mapper.k);

			if (knnNodeId >= nNodes)
				return;

			int cooPos = rptr[iRow];

			Tbx::Dual_quat_cu dqi, dqj;
			Tbx::Vec3 ri, ti, rj, tj;
			get_twist(mapper.nodeId, ri, ti);
			get_twist(knnNodeId, rj, tj);
			dqi.from_twist(ri, ti);
			dqj.from_twist(rj, tj);

			float4 nodeVwi = get_nodesVw(mapper.nodeId);
			float4 nodeVwj = get_nodesVw(knnNodeId);
			Tbx::Point3 vi(convert(read_float3_4(nodeVwi)));
			Tbx::Point3 vj(convert(read_float3_4(nodeVwj)));
			float alpha_ij = max(1.f / nodeVwi.w, 1.f / nodeVwj.w);
			float ww = sqrt(lambda * alpha_ij);

			// energy=============================================
			Tbx::Vec3 val = dqi.transform(Tbx::Point3(vj)) - dqj.transform(Tbx::Point3(vj));
			val = reg_term_penalty(val);

			fptr[iRow + 0] = val.x * ww;
			fptr[iRow + 1] = val.y * ww;
			fptr[iRow + 2] = val.z * ww;

			Tbx::Vec3 val1 = dqi.transform(Tbx::Point3(vi)) - dqj.transform(Tbx::Point3(vi));
			val1 = reg_term_penalty(val1);
			fptr[iRow + 3] = val1.x * ww;
			fptr[iRow + 4] = val1.y * ww;
			fptr[iRow + 5] = val1.z * ww;

			// jacobi=============================================
			for (int ialpha = 0; ialpha < VarPerNode; ialpha++)
			{
				Tbx::Transfo p_Ti_p_alpha = p_SE3_p_alpha_func(dqi, ialpha);
				Tbx::Transfo p_Tj_p_alpha = p_SE3_p_alpha_func(dqj, ialpha);

				// partial_psi_partial_alpha
				Tbx::Vec3 p_psi_p_alphai_j = (p_Ti_p_alpha * vj) * ww;
				Tbx::Vec3 p_psi_p_alphaj_j = (p_Tj_p_alpha * vj) * (-ww);
				Tbx::Vec3 p_psi_p_alphai_i = (p_Ti_p_alpha * vi) * ww;
				Tbx::Vec3 p_psi_p_alphaj_i = (p_Tj_p_alpha * vi) * (-ww);

				for (int ixyz = 0; ixyz < 3; ixyz++)
				{
					int pos = cooPos + ixyz*ColPerRow + ialpha;
					vptr[pos] = p_psi_p_alphai_j[ixyz];
					vptr[pos + VarPerNode] = p_psi_p_alphaj_j[ixyz];
					pos += 3 * ColPerRow;
					vptr[pos] = p_psi_p_alphai_i[ixyz];
					vptr[pos + VarPerNode] = p_psi_p_alphaj_i[ixyz];
				}
			}// end for ialpha
		}// end function ()
	};

	__global__ void calcRegTerm_kernel(RegTermJacobi rj)
	{
		rj();
	}

	void GpuGaussNewtonSolver::calcRegTerm()
	{
		RegTermJacobi rj;
		rj.cidx = m_Jr_ColIdx.ptr();
		rj.lambda = m_param->fusion_lambda;
		rj.nNodes = m_numNodes;
		rj.nRows = m_Jrrows;
		rj.psi_reg = m_param->fusion_psi_reg;
		rj.rows2nodeIds = m_Jr_RowMap2NodeId;
		rj.rptr = m_Jr_RowPtr.ptr();
		rj.vptr = m_Jr_val.ptr();
		rj.fptr = m_f_r.ptr();

		dim3 block(CTA_SIZE);
		dim3 grid(divUp(m_Jrrows / 6, block.x));

		calcRegTerm_kernel << <grid, block >> >(rj);
		cudaSafeCall(cudaGetLastError(), "calcRegTerm_kernel");

		// 2. compute Jrt ==============================================
		// 2.1. fill (row, col) as (col, row) from Jr and sort.
		cudaMemcpy(m_Jrt_RowPtr_coo.ptr(), m_Jr_ColIdx.ptr(), m_Jrnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(m_Jrt_val.ptr(), m_Jr_val.ptr(), m_Jrnnzs*sizeof(float), cudaMemcpyDeviceToDevice);
		modergpu_wrapper::mergesort_by_key(m_Jrt_RowPtr_coo.ptr(), m_Jrt_val.ptr(), m_Jrnnzs);
		cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcRegTerm::mergesort_by_key");
	}
#pragma endregion

#pragma region --calc Hessian
#define ENABLE_GPU_DUMP_DEBUG_H
	__global__ void calcJr0tJr0_add_to_Hd_kernel(float* Hd, int nLv0Nodes, const int* Jrt_rptr)
	{
		enum
		{
			VarPerNode = GpuGaussNewtonSolver::VarPerNode,
			VarPerNode2 = VarPerNode*VarPerNode,
			LowerPartNum = GpuGaussNewtonSolver::LowerPartNum
		};

		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		int iNode = tid / LowerPartNum;
		if (iNode >= nLv0Nodes)
			return;
		int eleLowerShift = tid - iNode*LowerPartNum;
		int rowShift = g_lower_2_rowShift_6x6[eleLowerShift];
		int colShift = g_lower_2_colShift_6x6[eleLowerShift];
		int row0 = iNode*VarPerNode;

		const int row0_begin = Jrt_rptr[row0 + rowShift];
		const int row_len = Jrt_rptr[row0 + rowShift + 1] - row0_begin;
		const int row1_begin = Jrt_rptr[row0 + colShift];

		float sum = 0.f;
		for (int i = 0; i < row_len; i++)
			sum += get_JrtVal(row1_begin + i) * get_JrtVal(row0_begin + i);

		Hd[iNode * VarPerNode2 + g_lower_2_full_6x6[eleLowerShift]] += sum;
	}
	
	__global__ void calcB_kernel(
		float* B_val, const int* B_rptr_coo, const int* B_cidx, 
		int nBrows, int Bnnz, const int* Jrt_rptr)
	{
		enum{VarPerNode = GpuGaussNewtonSolver::VarPerNode};

		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= Bnnz)
			return;

		int iBrow = B_rptr_coo[tid];
		int iBcol = B_cidx[tid];

		int Jr0t_cb = Jrt_rptr[iBrow];
		int Jr0t_ce = Jrt_rptr[iBrow + 1];

		int Jr1_rb = Jrt_rptr[iBcol + nBrows];
		int Jr1_re = Jrt_rptr[iBcol + nBrows + 1];

		float sum = 0.f;
		for (int i0 = Jr0t_cb, i1 = Jr1_rb; i0 < Jr0t_ce && i1 < Jr1_re; )
		{
			int Jr0t_c = get_JrtCidx(i0);
			int Jr1_r = get_JrtCidx(i1);
			if (Jr0t_c == Jr1_r)
			{
				for (int k = 0; k < VarPerNode; k++)
					sum += get_JrtVal(i0 + k) * get_JrtVal(i1 + k);
				i0 += VarPerNode;
				i1 += VarPerNode;
			}

			i0 += (Jr0t_c < Jr1_r) * VarPerNode;
			i1 += (Jr0t_c > Jr1_r) * VarPerNode;
		}// i

		B_val[tid] = sum;
	}

	void GpuGaussNewtonSolver::calcHessian()
	{
		// 1. compute Jr0'Jr0 and accumulate into Hd
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_numLv0Nodes*LowerPartNum, block.x));
			calcJr0tJr0_add_to_Hd_kernel << <grid, block >> >(m_Hd, m_numLv0Nodes, 
				m_Jrt_RowPtr.ptr());
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calcJr0tJr0_add_to_Hd_kernel");
		}

		// 2. compute B = Jr0'Jr1
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_Bnnzs, block.x));
			calcB_kernel << <grid, block >> >(m_B_val.ptr(), m_B_RowPtr_coo.ptr(), 
				m_B_ColIdx.ptr(), m_Brows, m_Bnnzs, m_Jrt_RowPtr.ptr());
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calcB_kernel");
		}

		// 3. compute Bt
		cudaMemcpy(m_Bt_RowPtr_coo.ptr(), m_B_ColIdx.ptr(), m_Bnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(m_Bt_val.ptr(), m_B_val.ptr(), m_Bnnzs*sizeof(float), cudaMemcpyDeviceToDevice);
		modergpu_wrapper::mergesort_by_key(m_Bt_RowPtr_coo.ptr(), m_Bt_val.ptr(), m_Bnnzs);
		cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::mergesort_by_key");
	}
#pragma endregion
}