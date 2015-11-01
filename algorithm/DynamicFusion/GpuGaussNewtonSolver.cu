#include "GpuGaussNewtonSolver.h"
#include "device_utils.h"
#include "cudpp\thrust_wrapper.h"
#include "cudpp\ModerGpuWrapper.h"
#include <iostream>
#include "GpuCholeSky.h"
namespace dfusion
{
//#define DEFINE_USE_HALF_GRAPH_EDGE
//#define CALC_DATA_TERM_NUMERIC
//#define CALC_REG_TERM_NUMERIC
//#define DEBUG_ASSIGN_10M_TO_NO_CORR
//#define DEBUG_ASSIGN_BIG_ENERGY_TO_NO_CORR

#ifdef DEFINE_USE_HALF_GRAPH_EDGE
	enum{RowPerNode_RegTerm = 3};
#else
	enum{ RowPerNode_RegTerm = 6 };
#endif
//#define USE_L2_NORM_DATA_TERM
//#define USE_L2_NORM_REG_TERM
#define CHECK(a, msg){if(!(a)) throw std::exception(msg);} 
#define CHECK_LE(a, b){if((a) > (b)) {std::cout << "" << #a << "(" << a << ")<=" << #b << "(" << b << ")";throw std::exception(" ###error!");}} 

	texture<WarpField::KnnIdx, cudaTextureType1D, cudaReadModeElementType> g_nodesKnnTex;
	texture<float4, cudaTextureType1D, cudaReadModeElementType> g_nodesVwTex;
	texture<float, cudaTextureType1D, cudaReadModeElementType> g_twistTex;
	texture<float, cudaTextureType1D, cudaReadModeElementType> g_JrtValTex;
	texture<int, cudaTextureType1D, cudaReadModeElementType> g_JrtCidxTex;
	texture<float, cudaTextureType1D, cudaReadModeElementType> g_BtValTex;
	texture<int, cudaTextureType1D, cudaReadModeElementType> g_BtCidxTex;
	texture<float, cudaTextureType1D, cudaReadModeElementType> g_BtLtinvValTex;
	texture<float, cudaTextureType1D, cudaReadModeElementType> g_HdLinvTex;

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

	__device__ __forceinline__ float get_BtVal(int i)
	{
		return tex1Dfetch(g_BtValTex, i);
	}
	__device__ __forceinline__ int get_BtCidx(int i)
	{
		return tex1Dfetch(g_BtCidxTex, i);
	}

	__device__ __forceinline__ float get_HdLinv(int i)
	{
		return tex1Dfetch(g_HdLinvTex, i);
	}

	__device__ __forceinline__ float get_BtLtinvVal(int i)
	{
		return tex1Dfetch(g_BtLtinvValTex, i);
	}
	__device__ __forceinline__ int get_BtLtinvCidx(int i)
	{
		return tex1Dfetch(g_BtCidxTex, i);
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
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error1!");
		}
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
			cudaBindTexture(&offset, &g_nodesVwTex, m_nodesVw.ptr(), &desc,
				m_nodesVw.size() * sizeof(float4));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error2!");
		}
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cudaBindTexture(&offset, &g_twistTex, m_twist.ptr(), &desc,
				m_twist.size() * sizeof(float));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error3!");
		}
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cudaBindTexture(&offset, &g_JrtValTex, m_Jrt_val.ptr(), &desc,
				m_Jrt_val.size() * sizeof(float));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error4!");
		}
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
			cudaBindTexture(&offset, &g_JrtCidxTex, m_Jrt_ColIdx.ptr(), &desc,
				m_Jrt_ColIdx.size() * sizeof(int));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error5!");
		}
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cudaBindTexture(&offset, &g_BtLtinvValTex, m_Bt_Ltinv_val.ptr(), &desc,
				m_Bt_Ltinv_val.size() * sizeof(float));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error6!");
		}
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
			cudaBindTexture(&offset, &g_BtCidxTex, m_Bt_ColIdx.ptr(), &desc,
				m_Bt_ColIdx.size() * sizeof(int));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error7!");
		}
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cudaBindTexture(&offset, &g_BtValTex, m_Bt_val.ptr(), &desc,
				m_Bt_val.size() * sizeof(float));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error8!");
		}
		if (1)
		{
			size_t offset;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cudaBindTexture(&offset, &g_HdLinvTex, m_Hd_Linv.ptr(), &desc,
				m_Hd_Linv.size() * sizeof(float));
			if (offset != 0)
				throw std::exception("GpuGaussNewtonSolver::bindTextures(): non-zero-offset error9!");
		}
	}

	void GpuGaussNewtonSolver::unBindTextures()
	{
		cudaUnbindTexture(g_twistTex);
		cudaUnbindTexture(g_nodesVwTex);
		cudaUnbindTexture(g_nodesKnnTex);
		cudaUnbindTexture(g_JrtValTex);
		cudaUnbindTexture(g_JrtCidxTex);
		cudaUnbindTexture(g_BtValTex);
		cudaUnbindTexture(g_BtCidxTex);
		cudaUnbindTexture(g_BtLtinvValTex);
		cudaUnbindTexture(g_HdLinvTex);
	}
#pragma endregion

#pragma region --calc data term

//#define ENABLE_GPU_DUMP_DEBUG

	//__device__ float g_totalEnergy;

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
		Tbx::Transfo Tlw_inv;
		Tbx::Transfo Tlw;

		int imgWidth;
		int imgHeight;
		int nNodes;

		float distThres;
		float angleThres;
		float psi_data;

		float* totalEnergy;


#ifdef ENABLE_GPU_DUMP_DEBUG
		// for debug
		float* debug_buffer_pixel_sum2;
		float* debug_buffer_pixel_val;
#endif

		__device__ __forceinline__ float data_term_energy(float f)const
		{
#ifdef USE_L2_NORM_DATA_TERM
			return 0.5f*f*f;
#else
			// the robust Tukey penelty gradient
			if (abs(f) <= psi_data)
				return psi_data*psi_data / 6.f *(1 - pow(1 - sqr(f / psi_data), 3));
			else
				return psi_data*psi_data / 6.f;
#endif
		}

		__device__ __forceinline__ float data_term_penalty(float f)const
		{
#ifdef USE_L2_NORM_DATA_TERM
			return f;
#else
			return f * sqr(max(0.f, 1.f - sqr(f / psi_data)));
			//// the robust Tukey penelty gradient
			//if (abs(f) <= psi_data)
			//	return f * sqr(1 - sqr(f / psi_data));
			//else
			//	return 0;
#endif
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
			Tbx::Vec3 Tn = dq.rotate(n);
			Tbx::Point3 Tv(dq.transform(v) - vl);
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

			return p_T_p_alphak;
		}

		__device__ __forceinline__ bool search(int x, int y, Tbx::Point3& vl) const
		{
			float3 vwarp = read_float3_4(vmap_warp(y, x));
			float3 nwarp = read_float3_4(nmap_warp(y, x));

			return search(vwarp, nwarp, vl);
		}

		__device__ __forceinline__ bool search(float3 vwarp, float3 nwarp, Tbx::Point3& vl) const
		{
			if (isnan(nwarp.x) || isnan(vwarp.x))
				return false;

			float3 uvd = intr.xyz2uvd(vwarp);
			int2 ukr = make_int2(__float2int_rn(uvd.x), __float2int_rn(uvd.y));

			// we use opengl coordinate, thus world.z should < 0
			if (ukr.x < 0 || ukr.y < 0 || ukr.x >= imgWidth || ukr.y >= imgHeight || vwarp.z >= 0)
				return false;

			float3 vlive = read_float3_4(vmap_live[ukr.y*imgWidth + ukr.x]);
			float3 nlive = read_float3_4(nmap_live[ukr.y*imgWidth + ukr.x]);
			if (isnan(nlive.x) || isnan(vlive.x))
				return false;

#ifndef DEBUG_ASSIGN_10M_TO_NO_CORR
			float dist = norm(vwarp - vlive);
			if (!(dist <= distThres))
				return false;

			float sine = norm(cross(nwarp, nlive));
			if (!(sine < angleThres))
				return false;
#endif

			vl = Tbx::Point3(vlive.x, vlive.y, vlive.z);

			return true;
		}

		template<int maxK>
		__device__ __forceinline__ void calc_dataterm () const
		{
			const int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
			const int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

			Tbx::Point3 vl;
			bool found_coresp = false;
			if (x < imgWidth && y < imgHeight)
				found_coresp = search(x, y, vl);

			vl = Tlw_inv * vl;

			if (found_coresp)
			{
				Tbx::Point3 v(convert(read_float3_4(vmap_cano(y, x))));
				Tbx::Vec3 n(convert(read_float3_4(nmap_cano(y, x))));

				const KnnIdx knn = vmapKnn(y, x);
				Tbx::Dual_quat_cu dq(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
				Tbx::Dual_quat_cu dqk_0;
				float wk[maxK];

				// dqk_0
				{
					Tbx::Vec3 r, t;
					get_twist(knn_k(knn, 0), r, t);
					float4 nodeVw = get_nodesVw(knn_k(knn, 0));
					Tbx::Vec3 nodesV(convert(read_float3_4(nodeVw)) - v);
					dqk_0.from_twist(r, t);
					float expIn = nodesV.dot(nodesV) * nodeVw.w * nodeVw.w;
					wk[0] = __expf(-0.5f * expIn);
					dq = dq + dqk_0 * wk[0];
				}

				// other dqk_k
#pragma unroll
				for (int k = 1; k < maxK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					
					Tbx::Vec3 r, t;
					get_twist(knnNodeId, r, t);
					float4 nodeVw = get_nodesVw(knnNodeId);
					Tbx::Vec3 nodesV(convert(read_float3_4(nodeVw))-v);
					Tbx::Dual_quat_cu dqk_k;
					dqk_k.from_twist(r, t);
					wk[k] = __expf(-0.5f * nodesV.dot(nodesV) * nodeVw.w * nodeVw.w)
						 *sign(dqk_0.get_non_dual_part().dot(dqk_k.get_non_dual_part()));
					dq = dq + dqk_k * wk[k];		
				}

				Tbx::Dual_quat_cu dq_bar = dq;
				float norm_dq_bar = dq_bar.norm();
				if (norm_dq_bar < Tbx::Dual_quat_cu::epsilon())
					return;
				float inv_norm_dq_bar = 1.f / norm_dq_bar;

				dq = dq * inv_norm_dq_bar; // normalize

				// the grad energy f
				const float f = data_term_penalty(dq.rotate(n).dot(dq.transform(v) - vl));

				// paitial_f_partial_T
				const Tbx::Transfo p_f_p_T = compute_p_f_p_T(n, v, vl, dq);

				for (int knnK = 0; knnK < maxK; knnK++)
				{
					float p_f_p_alpha[VarPerNode];
					int knnNodeId = knn_k(knn, knnK);
					float wk_k = wk[knnK] * inv_norm_dq_bar * 2;
					
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
				}// end for knnK
			}// end if found corr
		}// end function ()

		template<int maxK>
		__device__ __forceinline__ Tbx::Dual_quat_cu calc_pixel_dq(WarpField::KnnIdx knn, 
			Tbx::Point3 v, float* wk)const
		{
			Tbx::Dual_quat_cu dqk_0;
			Tbx::Dual_quat_cu dq(Tbx::Quat_cu(0,0,0,0), Tbx::Quat_cu(0,0,0,0));
			// dqk_0
			{
				Tbx::Vec3 r, t;
				get_twist(knn_k(knn, 0), r, t);
				float4 nodeVw = get_nodesVw(knn_k(knn, 0));
				Tbx::Vec3 nodesV(convert(read_float3_4(nodeVw)) - v);
				dqk_0.from_twist(r, t);
				wk[0] = __expf(-0.5f * nodesV.dot(nodesV) * nodeVw.w * nodeVw.w);
				dq += dqk_0 * wk[0];
			}

			// other dqk_k
#pragma unroll
			for (int k = 1; k < maxK; k++)
			{
				float4 nodeVw = get_nodesVw(knn_k(knn, k));
				Tbx::Vec3 nodesV(convert(read_float3_4(nodeVw)) - v);
				Tbx::Dual_quat_cu dqk_k;
				Tbx::Vec3 r, t;
				get_twist(knn_k(knn, k), r, t);
				dqk_k.from_twist(r, t);
				wk[k] = __expf(-0.5f * nodesV.dot(nodesV) * nodeVw.w * nodeVw.w)
					*sign(dqk_0.get_non_dual_part().dot(dqk_k.get_non_dual_part()));
				dq += dqk_k * wk[k];
			}
			return dq;
		}

		__device__ __forceinline__ void exchange_ri_k(WarpField::KnnIdx knn, 
			const float* wk, int k, int i, Tbx::Dual_quat_cu& dq, float& inc)const
		{
			Tbx::Vec3 r, t;
			get_twist(knn_k(knn, k), r, t);
			Tbx::Dual_quat_cu old_dqk, new_dqk;
			old_dqk.from_twist(r, t);
			inc = get_numeric_inc(r[i]);
			r[i] += inc;
			new_dqk.from_twist(r, t);
			dq -= old_dqk * wk[k];
			dq += new_dqk * wk[k] * sign(old_dqk.get_non_dual_part().dot(new_dqk.get_non_dual_part()));
		}
		__device__ __forceinline__ void exchange_ti_k(WarpField::KnnIdx knn,
			const float* wk, int k, int i, Tbx::Dual_quat_cu& dq, float& inc)const
		{
			Tbx::Vec3 r, t;
			get_twist(knn_k(knn, k), r, t);
			Tbx::Dual_quat_cu old_dqk, new_dqk;
			old_dqk.from_twist(r, t);
			inc = get_numeric_inc(t[i]);
			t[i] += inc;
			new_dqk.from_twist(r, t);
			dq -= old_dqk * wk[k];
			dq += new_dqk * wk[k] * sign(old_dqk.get_non_dual_part().dot(new_dqk.get_non_dual_part()));
		}

		__device__ __forceinline__ float get_numeric_inc(float v) const
		{
			return max( 1e-5f, v* 1e-3f);
		}

		template<int maxK>
		__device__ __forceinline__ void calc_dataterm_numeric() const
		{
			const int x = threadIdx.x + blockIdx.x * blockDim.x;
			const int y = threadIdx.y + blockIdx.y * blockDim.y;
			if (x >= imgWidth || y >= imgHeight)
				return;

			const KnnIdx knn = vmapKnn(y, x);
			Tbx::Point3 v(convert(read_float3_4(vmap_cano(y, x))));
			Tbx::Vec3 n(convert(read_float3_4(nmap_cano(y, x))));

			if (isnan(n.x) || isnan(v.x))
				return;

			// 1. get all nodes params
			// 2. compute function=================================================
			float wk[maxK];
			Tbx::Dual_quat_cu dq = calc_pixel_dq<maxK>(knn, v, wk);
			float norm_dq = dq.norm();
			if (norm_dq < Tbx::Dual_quat_cu::epsilon())
				return;
			Tbx::Dual_quat_cu dq_not_normalized = dq;
			dq = dq * (1.f / norm_dq); // normalize

			// find corr
			Tbx::Vec3 nwarp = Tlw*dq.rotate(n);
			Tbx::Point3 vwarp = Tlw*dq.transform(v);
			Tbx::Point3 vl;
			//bool corr_found = search(convert(vwarp), convert(nwarp), vl);
			bool corr_found = search(x, y, vl);
			if (!corr_found)
				return;

			// the grad energy
			const float f = nwarp.dot(vwarp - vl);
			const float psi_f = data_term_penalty(f);

			// 3. compute jacobi
			for (int knnK = 0; knnK < maxK; knnK++)
			{
				float df[6];

				// 3.0 p_r[0:2]
				for (int i = 0; i < 3; i++)
				{
					float inc;
					Tbx::Dual_quat_cu dq1 = dq_not_normalized;
					exchange_ri_k(knn, wk, knnK, i, dq1, inc);
					dq1 *= (1.f / dq1.norm());
					nwarp = Tlw*dq1.rotate(n);
					vwarp = Tlw*dq1.transform(v);

					Tbx::Point3 vl1 = vl;
					//corr_found = search(convert(vwarp), convert(nwarp), vl1);
					//if (!corr_found)
					//	return;

					float f1 = nwarp.dot(vwarp - vl1);
					df[i] = (f1 - f) / inc;
				}// i=0:3

				// 3.1 p_t[0:2]
				for (int i = 0; i < 3; i++)
				{
					float inc;
					Tbx::Dual_quat_cu dq1 = dq_not_normalized;
					exchange_ti_k(knn, wk, knnK, i, dq1, inc);
					dq1 *= (1.f / dq1.norm());
					nwarp = Tlw*dq1.rotate(n);
					vwarp = Tlw*dq1.transform(v);

					Tbx::Point3 vl1 = vl;
					//corr_found = search(convert(vwarp), convert(nwarp), vl1);
					//if (!corr_found)
					//	return;

					float f1 = nwarp.dot(vwarp - vl1);
					df[i+3] = (f1 - f) / inc;
				}// i=0:3

				//// reduce--------------------------------------------------
				int shift = knn_k(knn, knnK) * VarPerNode2;
				int shift_g = knn_k(knn, knnK) * VarPerNode;
				for (int i = 0; i < VarPerNode; ++i)
				{
#pragma unroll
					for (int j = 0; j <= i; ++j)
						atomicAdd(&Hd_[shift + j], df[i] * df[j]);
					atomicAdd(&g_[shift_g + i], df[i] * psi_f);
					shift += VarPerNode;
				}// end for i
			}// end for knnK
		}// end function ()

		template<int maxK>
		__device__ __forceinline__ void calcTotalEnergy()const
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
				float wk[maxK];
				// dqk_0
				{
					Tbx::Vec3 r, t;
					get_twist(knn_k(knn, 0), r, t);
					float4 nodeVw = get_nodesVw(knn_k(knn, 0));
					Tbx::Vec3 nodesV(convert(read_float3_4(nodeVw)) - v);
					dqk_0.from_twist(r, t);
					float expIn = nodesV.dot(nodesV) * nodeVw.w * nodeVw.w;
					wk[0] = __expf(-0.5f * expIn);
					dq = dq + dqk_0 * wk[0];
				}

				// other dqk_k
#pragma unroll
				for (int k = 1; k < maxK; k++)
				{
					int knnNodeId = knn_k(knn, k);

					Tbx::Vec3 r, t;
					get_twist(knnNodeId, r, t);
					float4 nodeVw = get_nodesVw(knnNodeId);
					Tbx::Vec3 nodesV(convert(read_float3_4(nodeVw)) - v);
					Tbx::Dual_quat_cu dqk_k;
					dqk_k.from_twist(r, t);
					wk[k] = __expf(-0.5f * nodesV.dot(nodesV) * nodeVw.w * nodeVw.w)
						*sign(dqk_0.get_non_dual_part().dot(dqk_k.get_non_dual_part()));
					dq = dq + dqk_k * wk[k];
				}

				float norm_dq = dq.norm();
				if (norm_dq < Tbx::Dual_quat_cu::epsilon())
					return;
				dq = dq * (1.f / norm_dq); // normalize

				// the grad energy f
				const float f = data_term_energy((Tlw*dq.rotate(n)).dot(Tlw*dq.transform(v) - vl));
				//atomicAdd(totalEnergy, f);
				totalEnergy[y*imgWidth + x] = f;
			}//end if find corr
#ifdef DEBUG_ASSIGN_BIG_ENERGY_TO_NO_CORR
			else // debug: add constant penalty
			{
				totalEnergy[y*imgWidth + x] = data_term_energy(psi_data);
			}
#endif
		}
	};

	template<int maxK>
	__global__ void dataTermCombinedKernel(const DataTermCombined cs)
	{
#ifdef CALC_DATA_TERM_NUMERIC
		cs.calc_dataterm_numeric<maxK>();
#else
		cs.calc_dataterm<maxK>();
#endif
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
		cs.Tlw_inv = m_pWarpField->get_rigidTransform().fast_invert();
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
		int maxK = min(WarpField::KnnK, m_pWarpField->getNumNodesInLevel(0));
		switch (maxK)
		{
		case 1:
			dataTermCombinedKernel<1> << <grid, block >> >(cs);
			break;
		case 2:
			dataTermCombinedKernel<2> << <grid, block >> >(cs);
			break;
		case 3:
			dataTermCombinedKernel<3> << <grid, block >> >(cs);
			break;
		case 4:
			dataTermCombinedKernel<4> << <grid, block >> >(cs);
			break;
		default:
			throw std::exception("non supported KnnK in data term calc!");
		}
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

	template<int maxK>
	__global__ void calcDataTermTotalEnergyKernel(const DataTermCombined cs)
	{
		cs.calcTotalEnergy<maxK>();
	}

#pragma endregion

#pragma region --define sparse structure
	__global__ void count_Jr_rows_kernel(int* rctptr, int nMaxNodes)
	{
		int i = threadIdx.x + blockIdx.x*blockDim.x;
		if (i >= nMaxNodes)
			return;
	
		WarpField::KnnIdx knn = get_nodesKnn(i);
		int numK = -1;
		for (int k = 0; k < WarpField::KnnK; ++k)
		{
			if (knn_k(knn, k) < nMaxNodes)
				numK = k;
		}

		// each node generate 6*maxK rows
		rctptr[i] = (numK + 1) * RowPerNode_RegTerm;
		
		if (i == 0)
			rctptr[nMaxNodes] = 0;
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
				mp.k = (r - row_b) / RowPerNode_RegTerm;
				mp.ixyz = r - RowPerNode_RegTerm * mp.k;
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
		if (iRow >= nRows)
			return;

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

		// the 1st thread also write the last value
		if (iRow == 0)
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
		if (m_Jrrows > 0)
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_numNodes, block.x));
			compute_row_map_kernel << <grid, block >> >(m_Jr_RowMap2NodeId.ptr(), m_Jr_RowCounter.ptr(), m_numNodes);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::initSparseStructure::compute_row_map_kernel");
		}
		if (m_Jrrows > 0)
		{
			CHECK_LE(m_Jrrows + 1, m_Jr_RowPtr.size());
			CHECK_LE(m_Jrcols + 1, m_Jrt_RowPtr.size());
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_Jrrows, block.x));
			compute_Jr_rowPtr_colIdx_kernel << <grid, block >> >(m_Jr_RowPtr.ptr(),
				m_Jr_RowPtr_coo.ptr(), m_Jr_ColIdx.ptr(), m_Jr_RowMap2NodeId.ptr(), m_numNodes, m_Jrrows);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::initSparseStructure::compute_Jr_rowPtr_kernel");
			cudaSafeCall(cudaMemcpy(&m_Jrnnzs, m_Jr_RowPtr.ptr() + m_Jrrows,
				sizeof(int), cudaMemcpyDeviceToHost), "copy Jr nnz to host");
			CHECK_LE(m_Jrnnzs, m_Jr_RowPtr_coo.size());
			CHECK_LE(0, m_Jrnnzs);
		}

		// 2. compute Jrt structure ==============================================
		// 2.1. fill (row, col) as (col, row) from Jr and sort.
		if (m_Jrrows > 0)
		{
			cudaMemcpy(m_Jrt_RowPtr_coo.ptr(), m_Jr_ColIdx.ptr(), m_Jrnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_Jrt_ColIdx.ptr(), m_Jr_RowPtr_coo.ptr(), m_Jrnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
			// !!!NOTE: we must use mergesort here, it can guarentees the order of values of the same key
			modergpu_wrapper::mergesort_by_key(m_Jrt_RowPtr_coo.ptr(), m_Jrt_ColIdx.ptr(), m_Jrnnzs);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::initSparseStructure::mergesort_by_key1");

			// 2.2. extract CSR rowptr info.
			if (CUSPARSE_STATUS_SUCCESS != cusparseXcoo2csr(m_cuSparseHandle,
				m_Jrt_RowPtr_coo.ptr(), m_Jrnnzs, m_Jrcols,
				m_Jrt_RowPtr.ptr(), CUSPARSE_INDEX_BASE_ZERO))
				throw std::exception("GpuGaussNewtonSolver::initSparseStructure::cusparseXcoo2csr failed");
		}

		// 3. compute B structure ==============================================
		// 3.1 the row ptr of B is the same with the first L0 rows of Jrt.
		if (m_Brows > 0)
		{
			CHECK_LE(m_Brows, m_B_RowPtr.size());
			CHECK_LE(m_Bcols, m_Bt_RowPtr.size());
			cudaMemcpy(m_B_RowPtr.ptr(), m_Jrt_RowPtr.ptr(), (m_Brows + 1)*sizeof(int), cudaMemcpyDeviceToDevice);
			cudaSafeCall(cudaMemcpy(&m_Bnnzs, m_B_RowPtr.ptr() + m_Brows,
				sizeof(int), cudaMemcpyDeviceToHost), "copy B nnz to host");
			CHECK_LE(m_Bnnzs, m_B_RowPtr_coo.size());
		}
		
		// 3.2 the col-idx of B
		if (m_Brows > 0)
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_Brows, block.x));
			calc_B_cidx_kernel << <grid, block >> >(m_B_RowPtr_coo.ptr(),
				m_B_ColIdx.ptr(), m_B_RowPtr.ptr(), m_Brows, m_numNodes, m_numLv0Nodes);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::initSparseStructure::calc_B_cidx_kernel");
		}

		// 3.3 sort to compute Bt
		if (m_Bnnzs > 0)
		{
			cudaMemcpy(m_Bt_RowPtr_coo.ptr(), m_B_ColIdx.ptr(), m_Bnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_Bt_ColIdx.ptr(), m_B_RowPtr_coo.ptr(), m_Bnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
			modergpu_wrapper::mergesort_by_key(m_Bt_RowPtr_coo.ptr(), m_Bt_ColIdx.ptr(), m_Bnnzs);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::initSparseStructure::mergesort_by_key2");
			if (CUSPARSE_STATUS_SUCCESS != cusparseXcoo2csr(m_cuSparseHandle,
				m_Bt_RowPtr_coo.ptr(), m_Bnnzs, m_Bcols,
				m_Bt_RowPtr.ptr(), CUSPARSE_INDEX_BASE_ZERO))
				throw std::exception("GpuGaussNewtonSolver::initSparseStructure::cusparseXcoo2csr failed");
		}
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

		float* totalEnergy;

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

		__device__ __forceinline__  float reg_term_energy(Tbx::Vec3 f)const
		{
#ifdef USE_L2_NORM_REG_TERM
			return 0.5f*f.dot(f);
#else
			// the robust Huber penelty gradient
			float s = 0;
			float norm = f.norm();
			if (norm < psi_reg)
				s = norm * norm * 0.5f;
			else
				s = psi_reg*(norm - psi_reg*0.5f);
			return s;
#endif
		}

		__device__ __forceinline__  Tbx::Vec3 reg_term_penalty(Tbx::Vec3 f)const
		{
#ifdef USE_L2_NORM_REG_TERM
			return f;
#else
			// the robust Huber penelty gradient
			Tbx::Vec3 df;
			float norm = f.norm();
			if (norm < psi_reg)
				df = f;
			else
			for (int k = 0; k < 3; k++)
				df[k] = f[k]*psi_reg / norm;
			return df;
#endif
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
			const int iRow = (threadIdx.x + blockIdx.x * blockDim.x)*RowPerNode_RegTerm;
		
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

			//if (isinf(nodeVwj.w))
			//	printf("inf found: %d %d %f %f %f %f\n", mapper.nodeId, knnNodeId, 
			//	nodeVwj.w, 1.f / nodeVwj.w, alpha_ij, ww);

			// debug
			//float ww2 = ww*ww;
			//ww = 1;
			//ww *= ww;

			// energy=============================================
			Tbx::Vec3 val = dqi.transform(Tbx::Point3(vj)) - dqj.transform(Tbx::Point3(vj));
			val = reg_term_penalty(val);

			fptr[iRow + 0] = val.x * ww;
			fptr[iRow + 1] = val.y * ww;
			fptr[iRow + 2] = val.z * ww;

#ifndef DEFINE_USE_HALF_GRAPH_EDGE
			Tbx::Vec3 val1 = dqj.transform(Tbx::Point3(vi)) - dqi.transform(Tbx::Point3(vi));
			val1 = reg_term_penalty(val1);
			fptr[iRow + 3] = val1.x * ww;
			fptr[iRow + 4] = val1.y * ww;
			fptr[iRow + 5] = val1.z * ww;
#endif

			// debug
			//ww = 1;
			//ww = ww2;

			// jacobi=============================================
			for (int ialpha = 0; ialpha < VarPerNode; ialpha++)
			{
				Tbx::Transfo p_Ti_p_alpha = p_SE3_p_alpha_func(dqi, ialpha);
				Tbx::Transfo p_Tj_p_alpha = p_SE3_p_alpha_func(dqj, ialpha);

				// partial_psi_partial_alpha
				Tbx::Vec3 p_psi_p_alphai_j = (p_Ti_p_alpha * vj) * ww;
				Tbx::Vec3 p_psi_p_alphaj_j = (p_Tj_p_alpha * vj) * (-ww);
#ifndef DEFINE_USE_HALF_GRAPH_EDGE
				Tbx::Vec3 p_psi_p_alphai_i = (p_Ti_p_alpha * vi) * (-ww);
				Tbx::Vec3 p_psi_p_alphaj_i = (p_Tj_p_alpha * vi) * ww;
#endif

				for (int ixyz = 0; ixyz < 3; ixyz++)
				{
					int pos = cooPos + ixyz*ColPerRow + ialpha;
					vptr[pos] = p_psi_p_alphai_j[ixyz];
					vptr[pos + VarPerNode] = p_psi_p_alphaj_j[ixyz];
#ifndef DEFINE_USE_HALF_GRAPH_EDGE
					pos += 3 * ColPerRow;
					vptr[pos] = p_psi_p_alphai_i[ixyz];
					vptr[pos + VarPerNode] = p_psi_p_alphaj_i[ixyz];
#endif
				}
			}// end for ialpha
		}// end function ()

		__device__ __forceinline__ float get_numeric_inc(float v) const
		{
			return max(1e-5f, v* 1e-3f);
		}

		__device__ __forceinline__ void calc_reg_numeric () const
		{
			const int iRow = (threadIdx.x + blockIdx.x * blockDim.x)*RowPerNode_RegTerm;

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
			Tbx::Vec3 val_j = dqi.transform(Tbx::Point3(vj)) - dqj.transform(Tbx::Point3(vj));
			Tbx::Vec3 psi_val_j = reg_term_penalty(val_j);

			fptr[iRow + 0] = psi_val_j.x * ww;
			fptr[iRow + 1] = psi_val_j.y * ww;
			fptr[iRow + 2] = psi_val_j.z * ww;

#ifndef DEFINE_USE_HALF_GRAPH_EDGE
			Tbx::Vec3 val_i = dqj.transform(Tbx::Point3(vi)) - dqi.transform(Tbx::Point3(vi));
			Tbx::Vec3 psi_val_i = reg_term_penalty(val_i);
			fptr[iRow + 3] = psi_val_i.x * ww;
			fptr[iRow + 4] = psi_val_i.y * ww;
			fptr[iRow + 5] = psi_val_i.z * ww;
#endif

			// jacobi=============================================
			for (int ialpha = 0; ialpha < 3; ialpha++)
			{
				float inci = get_numeric_inc(ri[ialpha]);
				ri[ialpha] += inci;
				dqi.from_twist(ri, ti);
				Tbx::Vec3 val_j_inci = dqi.transform(Tbx::Point3(vj)) - dqj.transform(Tbx::Point3(vj));
				Tbx::Vec3 val_i_inci = dqj.transform(Tbx::Point3(vi)) - dqi.transform(Tbx::Point3(vi));
				ri[ialpha] -= inci;
				dqi.from_twist(ri, ti);

				float incj = get_numeric_inc(rj[ialpha]);
				rj[ialpha] += incj;
				dqj.from_twist(rj, tj);
				Tbx::Vec3 val_j_incj = dqi.transform(Tbx::Point3(vj)) - dqj.transform(Tbx::Point3(vj));
				Tbx::Vec3 val_i_incj = dqj.transform(Tbx::Point3(vi)) - dqi.transform(Tbx::Point3(vi));
				rj[ialpha] -= incj;
				dqj.from_twist(rj, tj);

				for (int ixyz = 0; ixyz < 3; ixyz++)
				{
					int pos = cooPos + ixyz*ColPerRow + ialpha;
					vptr[pos] = ww * (val_j_inci[ixyz] - val_j[ixyz]) / inci;
					vptr[pos + VarPerNode] = ww * (val_j_incj[ixyz] - val_j[ixyz]) / incj;
#ifndef DEFINE_USE_HALF_GRAPH_EDGE
					pos += 3 * ColPerRow;
					vptr[pos] = ww * (val_i_inci[ixyz] - val_i[ixyz]) / inci;
					vptr[pos + VarPerNode] = ww * (val_i_incj[ixyz] - val_i[ixyz]) / incj;
#endif
				}
			}// end for ialpha
			for (int ialpha = 0; ialpha < 3; ialpha++)
			{
				float inci = get_numeric_inc(ti[ialpha]);
				ti[ialpha] += inci;
				dqi.from_twist(ri, ti);
				Tbx::Vec3 val_j_inci = dqi.transform(Tbx::Point3(vj)) - dqj.transform(Tbx::Point3(vj));
				Tbx::Vec3 val_i_inci = dqj.transform(Tbx::Point3(vi)) - dqi.transform(Tbx::Point3(vi));
				ti[ialpha] -= inci;
				dqi.from_twist(ri, ti);

				float incj = get_numeric_inc(tj[ialpha]);
				tj[ialpha] += incj;
				dqj.from_twist(rj, tj);
				Tbx::Vec3 val_j_incj = dqi.transform(Tbx::Point3(vj)) - dqj.transform(Tbx::Point3(vj));
				Tbx::Vec3 val_i_incj = dqj.transform(Tbx::Point3(vi)) - dqi.transform(Tbx::Point3(vi));
				tj[ialpha] -= incj;
				dqj.from_twist(rj, tj);

				for (int ixyz = 0; ixyz < 3; ixyz++)
				{
					int pos = cooPos + ixyz*ColPerRow + ialpha + 3;
					vptr[pos] = ww * (val_j_inci[ixyz] - val_j[ixyz]) / inci;
					vptr[pos + VarPerNode] = ww * (val_j_incj[ixyz] - val_j[ixyz]) / incj;
#ifndef DEFINE_USE_HALF_GRAPH_EDGE
					pos += 3 * ColPerRow;
					vptr[pos] = ww * (val_i_inci[ixyz] - val_i[ixyz]) / inci;
					vptr[pos + VarPerNode] = ww * (val_i_incj[ixyz] - val_i[ixyz]) / incj;
#endif
				}
			}// end for ialpha
		}// end function ()

		__device__ __forceinline__ void calcTotalEnergy () const
		{
			const int iNode = threadIdx.x + blockIdx.x * blockDim.x;
			const int iRow = iNode * RowPerNode_RegTerm;

			if (iRow >= nRows)
				return;

			Mapper mapper = rows2nodeIds[iRow];
			int knnNodeId = knn_k(get_nodesKnn(mapper.nodeId), mapper.k);

			if (knnNodeId >= nNodes)
				return;

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
			float ww2 = lambda * alpha_ij;

			// energy=============================================
			Tbx::Vec3 val = dqi.transform(Tbx::Point3(vj)) - dqj.transform(Tbx::Point3(vj));
			float eg = ww2 * reg_term_energy(val);
#ifndef DEFINE_USE_HALF_GRAPH_EDGE
			Tbx::Vec3 val1 = dqi.transform(Tbx::Point3(vi)) - dqj.transform(Tbx::Point3(vi));
			eg += ww2 * reg_term_energy(val1);
#endif

			//atomicAdd(totalEnergy, eg);
			totalEnergy[iNode] = eg;
		}
	};

	__global__ void calcRegTerm_kernel(RegTermJacobi rj)
	{
#ifdef CALC_REG_TERM_NUMERIC
		rj.calc_reg_numeric();
#else
		rj();
#endif
	}
	__global__ void calcRegTermTotalEnergy_kernel(RegTermJacobi rj)
	{
		rj.calcTotalEnergy();
	}

	void GpuGaussNewtonSolver::calcRegTerm()
	{
		if (m_Jrrows > 0)
		{
			CHECK_LE(1, m_Jrrows);
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
			dim3 grid(divUp(m_Jrrows / RowPerNode_RegTerm, block.x));

			calcRegTerm_kernel << <grid, block >> >(rj);
			cudaSafeCall(cudaGetLastError(), "calcRegTerm_kernel");

			// 2. compute Jrt ==============================================
			// 2.1. fill (row, col) as (col, row) from Jr and sort.
			cudaMemcpy(m_Jrt_RowPtr_coo.ptr(), m_Jr_ColIdx.ptr(), m_Jrnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_Jrt_val.ptr(), m_Jr_val.ptr(), m_Jrnnzs*sizeof(float), cudaMemcpyDeviceToDevice);
			modergpu_wrapper::mergesort_by_key(m_Jrt_RowPtr_coo.ptr(), m_Jrt_val.ptr(), m_Jrnnzs);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcRegTerm::mergesort_by_key");
		}
	}
#pragma endregion

#pragma region --calc Hessian
#define ENABLE_GPU_DUMP_DEBUG_H
	__global__ void calcJr0tJr0_add_to_Hd_kernel(float* Hd, int nLv0Nodes, 
		const int* Jrt_rptr, float diag_eps)
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

		float sum = diag_eps * (rowShift == colShift);
		for (int i = 0; i < row_len; i++)
			sum += get_JrtVal(row1_begin + i) * get_JrtVal(row0_begin + i);

		Hd[iNode * VarPerNode2 + rowShift*VarPerNode+colShift] += sum;
	}
	
	__global__ void fill_Hd_upper_kernel(float* Hd, int nLv0Nodes)
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
		
		Hd[iNode * VarPerNode2 + colShift * VarPerNode + rowShift] = 
			Hd[iNode * VarPerNode2 + rowShift * VarPerNode + colShift];
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

	__global__ void calcHr_kernel(float* Hr, const int* Jrt_rptr,
		int HrRowsCols, int nBrows, float diag_eps)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= (HrRowsCols + 1)*HrRowsCols / 2)
			return;

		// y is the triangular number
		int y = floor(-0.5 + sqrt(0.25 + 2 * tid));
		int triangularNumber = y * (y + 1) / 2;
		// x should <= y
		int x = tid - triangularNumber;

		int Jrt13_ib = Jrt_rptr[y + nBrows];
		int Jrt13_ie = Jrt_rptr[y + nBrows + 1];
		int Jrt13_jb = Jrt_rptr[x + nBrows];
		int Jrt13_je = Jrt_rptr[x + nBrows + 1];

		float sum = diag_eps * (x == y);
		for (int i = Jrt13_ib, j = Jrt13_jb; i < Jrt13_ie && j < Jrt13_je;)
		{
			int ci = get_JrtCidx(i);
			int cj = get_JrtCidx(j);
			if (ci == cj)
			{
				float s = 0.f;
				for (int k = 0; k < GpuGaussNewtonSolver::VarPerNode; k++)
					s += get_JrtVal(i + k) * get_JrtVal(j + k);
				sum += s;
				i += GpuGaussNewtonSolver::VarPerNode;
				j += GpuGaussNewtonSolver::VarPerNode;
			}

			i += (ci < cj) * GpuGaussNewtonSolver::VarPerNode;
			j += (ci > cj) * GpuGaussNewtonSolver::VarPerNode;
		}// i

		Hr[y*HrRowsCols + x] = Hr[x*HrRowsCols + y] = sum;
	}

	void GpuGaussNewtonSolver::calcHessian()
	{
		// 1. compute Jr0'Jr0 and accumulate into Hd
		if (m_Jrrows > 0)
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_numLv0Nodes*LowerPartNum, block.x));
			calcJr0tJr0_add_to_Hd_kernel << <grid, block >> >(m_Hd, m_numLv0Nodes, 
				m_Jrt_RowPtr.ptr(), m_param->fusion_GaussNewton_diag_regTerm);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calcJr0tJr0_add_to_Hd_kernel");
		}

		// 1.1 fill the upper tri part of Hd
		// previously, we only calculate the lower triangular pert of Hd;
		// now that the computation of Hd is ready, we fill the mission upper part
		if (m_numLv0Nodes > 0)
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_numLv0Nodes*LowerPartNum, block.x));
			fill_Hd_upper_kernel << <grid, block >> >(m_Hd, m_numLv0Nodes);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::fill_Hd_upper_kernel");
		}

		// 2. compute B = Jr0'Jr1
		if (m_Bnnzs > 0)
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_Bnnzs, block.x));
			calcB_kernel << <grid, block >> >(m_B_val.ptr(), m_B_RowPtr_coo.ptr(), 
				m_B_ColIdx.ptr(), m_Brows, m_Bnnzs, m_Jrt_RowPtr.ptr());
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calcB_kernel");
		}

		// 3. compute Bt
		if (m_Bnnzs > 0)
		{
			cudaMemcpy(m_Bt_RowPtr_coo.ptr(), m_B_ColIdx.ptr(), m_Bnnzs*sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_Bt_val.ptr(), m_B_val.ptr(), m_Bnnzs*sizeof(float), cudaMemcpyDeviceToDevice);
			modergpu_wrapper::mergesort_by_key(m_Bt_RowPtr_coo.ptr(), m_Bt_val.ptr(), m_Bnnzs);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::mergesort_by_key");
		}

		// 4. compute Hr
		CHECK_LE(m_HrRowsCols*m_HrRowsCols, m_Hr.size());
		if (m_HrRowsCols > 0)
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_HrRowsCols*(m_HrRowsCols+1)/2, block.x));
			calcHr_kernel << <grid, block >> >(m_Hr.ptr(), m_Jrt_RowPtr.ptr(),
				m_HrRowsCols, m_Brows, m_param->fusion_GaussNewton_diag_regTerm);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calcHr_kernel");
		}

		// 5. compute g = -(g + Jr'*fr)
		if (m_Jrrows > 0)
		{
			float alpha = -1.f;
			float beta = -1.f;
			if (CUSPARSE_STATUS_SUCCESS != cusparseScsrmv(
				m_cuSparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_Jrcols,
				m_Jrrows, m_Jrnnzs, &alpha, m_Jrt_desc, m_Jrt_val.ptr(), m_Jrt_RowPtr.ptr(),
				m_Jrt_ColIdx.ptr(), m_f_r.ptr(), &beta, m_g.ptr()))
				throw std::exception("GpuGaussNewtonSolver::calcHessian::cusparseScsrmv failed!\n");
		}
	}
#pragma endregion

#pragma region --block solve
	__global__ void calcBtLtinv_kernel(float* BtLtinv, const int* Bt_rptr, 
		const int* Bt_rptr_coo, int nLv0Nodes, int nnz)
	{
		enum{ VarPerNode = GpuGaussNewtonSolver::VarPerNode };
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= nnz)
			return;

		int row = Bt_rptr_coo[tid];
		int col = get_BtLtinvCidx(tid);
		int iNodeCol = col / VarPerNode;
		int cshift = col - iNodeCol * VarPerNode;

		float sum = 0.f;
		int Hd_row_b = iNodeCol * VarPerNode;
		int Bt_b = Bt_rptr[row] / VarPerNode;
		int Bt_e = Bt_rptr[row + 1] / VarPerNode;
		int Bt_col_b = -1;

		// binary search Hd_row_b in the range col of [Bt_b, Bt_e]
		while (Bt_b < Bt_e)
		{
			int imid = ((Bt_b + Bt_e) >> 1);
			Bt_col_b = get_BtCidx(imid*VarPerNode);
			if (Bt_col_b < Hd_row_b)
				Bt_b = imid + 1;
			else
				Bt_e = imid;
		}
		Bt_b *= VarPerNode;
		Bt_e *= VarPerNode;
		Bt_col_b = get_BtCidx(Bt_b);
		if (Bt_col_b == Hd_row_b && Bt_b == Bt_e)
		{
			Hd_row_b = (Hd_row_b + cshift) * VarPerNode;
			for (int k = 0; k <= cshift; k++)
				sum += get_BtVal(Bt_b + k) * get_HdLinv(Hd_row_b + k);
		}

		// write the result
		BtLtinv[tid] = sum;
	}

	__global__ void calcQ_kernel(float* Q, const float* Hr,
		const int* Bt_rptr, int HrRowsCols, int nBrows)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid >= (HrRowsCols + 1)*HrRowsCols / 2)
			return;

		// y is the triangular number
		int y = floor(-0.5 + sqrt(0.25 + 2 * tid));
		int triangularNumber = y * (y + 1) / 2;
		// x should <= y
		int x = tid - triangularNumber;

		int Bt_ib = Bt_rptr[y];
		int Bt_ie = Bt_rptr[y + 1];
		int Bt_jb = Bt_rptr[x];
		int Bt_je = Bt_rptr[x + 1];

		float sum = 0.f;
		for (int i = Bt_ib, j = Bt_jb; i < Bt_ie && j < Bt_je;)
		{
			int ci = get_BtLtinvCidx(i);
			int cj = get_BtLtinvCidx(j);
			if (ci == cj)
			{
				float s = 0.f;
				for (int k = 0; k < GpuGaussNewtonSolver::VarPerNode; k++)
					s += get_BtLtinvVal(i + k) * get_BtLtinvVal(j + k);
				sum += s;
				i += GpuGaussNewtonSolver::VarPerNode;
				j += GpuGaussNewtonSolver::VarPerNode;
			}

			i += (ci < cj) * GpuGaussNewtonSolver::VarPerNode;
			j += (ci > cj) * GpuGaussNewtonSolver::VarPerNode;
		}// i

		Q[y*HrRowsCols + x] = Q[x*HrRowsCols + y] = Hr[y*HrRowsCols + x] - sum;
	}

	// vec_out = alpha * Linv * vec_in + beta * vec_out
	__global__ void calc_Hd_Linv_x_vec_kernel(float* vec_out, const float* vec_in, int nRows,
		float alpha = 1.f, float beta = 0.f)
	{
		int iRow = threadIdx.x + blockIdx.x*blockDim.x;
		if (iRow >= nRows)
			return;
		int iNode = iRow / GpuGaussNewtonSolver::VarPerNode;
		int rshift = iRow - iNode * GpuGaussNewtonSolver::VarPerNode;
		int iPos = iRow * GpuGaussNewtonSolver::VarPerNode + rshift;

		float sum = 0.f;
		for (int k = -rshift; k <= 0; k++)
			sum += get_HdLinv(iPos + k) * vec_in[iRow + k];

		vec_out[iRow] = alpha * sum + beta * vec_out[iRow];
	}

	// vec_out = alpha * Ltinv * vec_in + beta * vec_out
	__global__ void calc_Hd_Ltinv_x_vec_kernel(float* vec_out, const float* vec_in, int nRows,
		float alpha = 1.f, float beta = 0.f)
	{
		int iRow = threadIdx.x + blockIdx.x*blockDim.x;
		if (iRow >= nRows)
			return;
		int iNode = iRow / GpuGaussNewtonSolver::VarPerNode;
		int rshift = iRow - iNode * GpuGaussNewtonSolver::VarPerNode;
		int iPos = iRow * GpuGaussNewtonSolver::VarPerNode + rshift;

		float sum = 0.f;
		for (int k = 0; k < GpuGaussNewtonSolver::VarPerNode-rshift; k++)
			sum += get_HdLinv(iPos + k*GpuGaussNewtonSolver::VarPerNode) * vec_in[iRow + k];

		vec_out[iRow] = alpha * sum + beta * vec_out[iRow];
	}

	void GpuGaussNewtonSolver::blockSolve()
	{
		cusparseStatus_t cuSparseStatus;
		CHECK_LE(m_numLv0Nodes*VarPerNode*VarPerNode, m_Hd_Linv.size());
		CHECK_LE(m_numLv0Nodes*VarPerNode*VarPerNode, m_Hd_LLtinv.size());

		// 1. batch LLt the diag blocks Hd==================================================

		// 1.0. copy Hd to Linv buffer
		cudaSafeCall(cudaMemcpy(m_Hd_Linv.ptr(), m_Hd.ptr(), m_numLv0Nodes*VarPerNode
			*VarPerNode*m_Hd.elem_size, cudaMemcpyDeviceToDevice), 
			"GpuGaussNewtonSolver::blockSolve::copy Hd to Hd_L");

		// 1.1 Hd = L*L'
		gpu_cholesky::single_thread_cholesky_batched(m_Hd_Linv.ptr(), VarPerNode,
			VarPerNode*VarPerNode, m_numLv0Nodes);
		checkNan(m_Hd_Linv, m_numLv0Nodes*VarPerNode*VarPerNode, "Hd_L");

		// 1.2 inv(L)
		gpu_cholesky::single_thread_tril_inv_batched(m_Hd_Linv.ptr(), VarPerNode,
			VarPerNode*VarPerNode, m_numLv0Nodes);
		checkNan(m_Hd_Linv, m_numLv0Nodes*VarPerNode*VarPerNode, "Hd_Linv");

		// 1.3 inv(L*L') = inv(L')*inv(L) = inv(L)'*inv(L)
		gpu_cholesky::single_thread_LtL_batched(
			m_Hd_LLtinv.ptr(), VarPerNode*VarPerNode, m_Hd_Linv.ptr(), 
			VarPerNode*VarPerNode, VarPerNode, m_numLv0Nodes);
		checkNan(m_Hd_LLtinv, m_numLv0Nodes*VarPerNode*VarPerNode, "Hd_LLtinv");

		// 2. compute Q = Hr - Bt * inv(Hd) * B ======================================
		CHECK_LE(m_HrRowsCols*m_HrRowsCols, m_Q.size());
		// 2.1 compute Bt*Ltinv
		if (m_HrRowsCols > 0)
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_Bnnzs, block.x));
			calcBtLtinv_kernel << <grid, block >> >(m_Bt_Ltinv_val.ptr(),
				m_Bt_RowPtr.ptr(), m_Bt_RowPtr_coo.ptr(), m_numLv0Nodes, m_Bnnzs);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calcBtLtinv_kernel");
		}

		// 2.2 compute Q
		if (m_HrRowsCols > 0)
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_HrRowsCols*(m_HrRowsCols+1)/2, block.x));
			calcQ_kernel << <grid, block >> >(m_Q.ptr(), m_Hr.ptr(), m_Bt_RowPtr.ptr(),
				m_HrRowsCols, m_Brows);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calcQ_kernel");

			// kept Q before factorize, for debug
			if (m_Q_kept.size() == m_Q.size())
				cudaMemcpy(m_Q_kept.ptr(), m_Q.ptr(),
					m_HrRowsCols*m_HrRowsCols*m_Q.elem_size,
					cudaMemcpyDeviceToDevice);
			checkNan(m_Q, m_HrRowsCols*m_HrRowsCols, "Q");
		}

		// 3. llt decompostion of Q ==================================================
		// 3.1 decide the working space of the solver
		if (m_HrRowsCols > 0)
		{
			int lwork = 0;
			cusolverDnSpotrf_bufferSize(m_cuSolverHandle, CUBLAS_FILL_MODE_LOWER,
				m_HrRowsCols, m_Q.ptr(), m_HrRowsCols, &lwork);
			if (lwork > m_cuSolverWorkSpace.size())
			{
				// we store dev info in the last element
				m_cuSolverWorkSpace.create(lwork * 1.5 + 1);
				printf("cusolverDnSpotrf_bufferSize: %d\n", lwork);
			}

			// 3.2 Cholesky decomposition
			// before this step, m_Q is calculated as filled as symmetric matrix
			// note that cublas uses column majored storage, thus after this step
			// the matrix m_Q should be viewed as column-majored matrix
			cusolverStatus_t fst = cusolverDnSpotrf(m_cuSolverHandle, CUBLAS_FILL_MODE_LOWER, m_HrRowsCols,
				m_Q.ptr(), m_HrRowsCols, m_cuSolverWorkSpace.ptr(), lwork,
				(int*)m_cuSolverWorkSpace.ptr() + m_cuSolverWorkSpace.size() - 1);
			if (CUSOLVER_STATUS_SUCCESS != fst)
			{
				printf("cusolverDnSpotrf failed: status: %d\n", fst);
				throw std::exception();
			}
			checkNan(m_Q, m_HrRowsCols*m_HrRowsCols, "Q1");
		}
		// 4. solve H*h = g =============================================================
		const int sz = m_Jrcols;
		const int sz0 = m_Brows;
		const int sz1 = sz - sz0;
		CHECK_LE(sz, m_u.size());
		CHECK_LE(sz, m_h.size());
		CHECK_LE(sz, m_g.size());
		CHECK_LE(sz, m_tmpvec.size());

		// 4.1 let H = LL', first we solve for L*u=g;
		// 4.1.1 u(0:sz0-1) = HdLinv*g(0:sz0-1)
		if (sz0 > 0)
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(sz0, block.x));
			calc_Hd_Linv_x_vec_kernel << <grid, block >> >(m_u.ptr(),
				m_g.ptr(), sz0);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calc_Hd_Linv_x_vec_kernel");
		}
	
		// 4.1.2 u(sz0:sz-1) = LQinv*(g(sz0:sz-1) - Bt*HdLtinv*HdLinv*g(0:sz0-1))
		if (sz1 > 0)
		{
			// tmpvec = HdLtinv*HdLinv*g(0:sz0-1)
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(sz0, block.x));
			calc_Hd_Ltinv_x_vec_kernel << <grid, block >> >(m_tmpvec.ptr(),
				m_u.ptr(), sz0);
			cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calc_Hd_Ltinv_x_vec_kernel");

			// u(sz0:sz-1) = g(sz0:sz-1) - Bt*tmpvec
			{
				float alpha = -1.f;
				float beta = 1.f;
				cudaMemcpy(m_u.ptr() + sz0, m_g.ptr() + sz0, sz1*sizeof(float), cudaMemcpyDeviceToDevice);
				cuSparseStatus = cusparseScsrmv(m_cuSparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, sz1, sz0,
					m_Bnnzs, &alpha, m_Bt_desc, m_Bt_val.ptr(), m_Bt_RowPtr.ptr(),
					m_Bt_ColIdx.ptr(), m_tmpvec.ptr(), &beta, m_u.ptr() + sz0);
				if (cuSparseStatus != CUSPARSE_STATUS_SUCCESS)
					printf("cuSparse error1: %d\n", cuSparseStatus);

				// solve LQ*u(sz0:sz-1) = u(sz0:sz-1)
				// note cublas use column majored matrix, we assume m_Q is column majored in this step
				cublasStrsv(m_cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
					sz1, m_Q.ptr(), sz1, m_u.ptr() + sz0, 1);
			}
		}
		checkNan(m_u, sz, "u");
		
		// 4.2 then we solve for L'*h=u;
		// 4.2.1 h(sz0:sz-1) = UQinv*u(sz0:sz-1)
		if (sz1 > 0)
		{
			cudaMemcpy(m_h.ptr() + sz0, m_u.ptr() + sz0, sz1*sizeof(float), cudaMemcpyDeviceToDevice);
			cublasStrsv(m_cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
				sz1, m_Q.ptr(), sz1, m_h.ptr() + sz0, 1);
		}
		
		// 4.2.2 h(0:sz0-1) = HdLtinv*( u(0:sz0-1) - HdLinv*B*h(sz0:sz-1) )
		// tmpvec = B*h(sz0:sz-1)
		if (sz1 > 0)
		{
			float alpha = 1.f;
			float beta = 0.f;
			cuSparseStatus = cusparseScsrmv(m_cuSparseHandle,
				CUSPARSE_OPERATION_NON_TRANSPOSE, sz0, sz1,
				m_Bnnzs, &alpha, m_B_desc, m_B_val.ptr(), m_B_RowPtr.ptr(),
				m_B_ColIdx.ptr(), m_h.ptr() + sz0, &beta, m_tmpvec.ptr());
			if (cuSparseStatus != CUSPARSE_STATUS_SUCCESS)
				printf("cuSparse error2: %d\n", cuSparseStatus);
		}

		// u(0:sz0-1) = u(0:sz0-1) - HdLinv * tmpvec
		// h(0:sz0-1) = HdLtinv*u(0:sz0-1)
		if (sz0 > 0)
		{
			dim3 block(CTA_SIZE);
			dim3 grid(divUp(sz0, block.x));
			if (sz1 > 0)
			{
				calc_Hd_Linv_x_vec_kernel << <grid, block >> >(m_u.ptr(),
					m_tmpvec.ptr(), sz0, -1.f, 1.f);
				cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calc_Hd_Linv_x_vec_kernel");
				calc_Hd_Ltinv_x_vec_kernel << <grid, block >> >(m_h.ptr(),
					m_u.ptr(), sz0);
				cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calc_Hd_Ltinv_x_vec_kernel");
			}
			else
			{
				calc_Hd_Ltinv_x_vec_kernel << <grid, block >> >(m_h.ptr(),
					m_u.ptr(), sz0, -1.f);
				cudaSafeCall(cudaGetLastError(), "GpuGaussNewtonSolver::calcHessian::calc_Hd_Ltinv_x_vec_kernel");
			}
		}
	}

	float GpuGaussNewtonSolver::calcTotalEnergy(float& data_energy, float& reg_energy)
	{
		float total_energy = 0.f;
		cudaMemset(m_energy_vec.ptr(), 0, m_energy_vec.sizeBytes());
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
			cs.Tlw_inv = m_pWarpField->get_rigidTransform().fast_invert();
			cs.psi_data = m_param->fusion_psi_data;
			cs.totalEnergy = m_energy_vec.ptr();

			//int zero_mem_symbol = 0;
			//cudaMemcpyToSymbol(g_totalEnergy, &zero_mem_symbol, sizeof(int));
			//cudaMemset(&m_tmpvec[0], 0, sizeof(float));

			// 1. data term
			//////////////////////////////
			dim3 block(CTA_SIZE_X, CTA_SIZE_Y);
			dim3 grid(1, 1, 1);
			grid.x = divUp(cs.imgWidth, block.x);
			grid.y = divUp(cs.imgHeight, block.y);
			int maxK = min(WarpField::KnnK, m_pWarpField->getNumNodesInLevel(0));
			switch (maxK)
			{
			case 1:
				calcDataTermTotalEnergyKernel<1> << <grid, block >> >(cs);
				break;
			case 2:
				calcDataTermTotalEnergyKernel<2> << <grid, block >> >(cs);
				break;
			case 3:
				calcDataTermTotalEnergyKernel<3> << <grid, block >> >(cs);
				break;
			case 4:
				calcDataTermTotalEnergyKernel<4> << <grid, block >> >(cs);
				break;
			default:
				throw std::exception("non supported KnnK in data term energy!");
			}
			cudaSafeCall(cudaGetLastError(), "calcDataTermTotalEnergyKernel");
		}

		if (m_Jrrows > 0)
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
			rj.totalEnergy = m_energy_vec.ptr() + m_vmapKnn.rows()*m_vmapKnn.cols();

			dim3 block(CTA_SIZE);
			dim3 grid(divUp(m_Jrrows / RowPerNode_RegTerm, block.x));

			calcRegTermTotalEnergy_kernel << <grid, block >> >(rj);
			cudaSafeCall(cudaGetLastError(), "calcRegTermTotalEnergy_kernel");
		}

		//cudaSafeCall(cudaMemcpy(&total_energy,
		//	&m_tmpvec[0], sizeof(float), cudaMemcpyDeviceToHost), "copy reg totalEnergy to host");
		cublasStatus_t st = cublasSasum(m_cublasHandle, m_Jrrows / RowPerNode_RegTerm + 
			m_vmapKnn.rows()*m_vmapKnn.cols(),
			m_energy_vec.ptr(), 1, &total_energy);
		if (st != CUBLAS_STATUS_SUCCESS)
			throw std::exception("cublass error, in cublasSnrm2");

		// debug get both data and reg term energy
#if 1
		reg_energy = 0.f;
		if (m_Jrrows > 0)
		{
			cublasSasum(m_cublasHandle, m_Jrrows / RowPerNode_RegTerm,
				m_energy_vec.ptr() + m_vmapKnn.rows()*m_vmapKnn.cols(),
				1, &reg_energy);
		}
		data_energy = total_energy - reg_energy;
#endif

		return total_energy;
	}
#pragma endregion

#pragma region --update twist

	__global__ void updateTwist_inch_kernel(float* twist, const float* h, float step, int nNodes)
	{
		int i = threadIdx.x + blockIdx.x*blockDim.x;
		if (i < nNodes)
		{
			int i6 = i * 6;
			Tbx::Vec3 r(twist[i6] + step*h[i6], twist[i6 + 1] + step*h[i6 + 1], twist[i6 + 2] + step*h[i6 + 2]);
			Tbx::Vec3 t(twist[i6+3] + step*h[i6+3], twist[i6 + 4] + step*h[i6 + 4], twist[i6 + 5] + step*h[i6 + 5]);
			Tbx::Dual_quat_cu dq;
			dq.from_twist(r, t);
			dq.to_twist(r, t);
			twist[i6] = r[0];
			twist[i6 + 1] = r[1];
			twist[i6 + 2] = r[2];
			twist[i6 + 3] = t[0];
			twist[i6 + 4] = t[1];
			twist[i6 + 5] = t[2];
		}
	}

	void GpuGaussNewtonSolver::updateTwist_inch(const float* h, float step)
	{
		dim3 block(CTA_SIZE);
		dim3 grid(divUp(m_numNodes, block.x));
		updateTwist_inch_kernel << <grid, block >> >(m_twist.ptr(), h, step, m_numNodes);
		cudaSafeCall(cudaGetLastError(), "updateTwist_inch_kernel");
	}
#pragma endregion

#pragma region --factor out rigid

	__device__ float _g_common_q[8];

	template<int CTA_SIZE_, typename T>
	static __device__ __forceinline__ void reduce(volatile T* buffer)
	{
		int tid = Block::flattenedThreadId();
		T val = buffer[tid];

		if (CTA_SIZE_ >= 1024) { if (tid < 512) buffer[tid] = val = val + buffer[tid + 512]; __syncthreads(); }
		if (CTA_SIZE_ >= 512) { if (tid < 256) buffer[tid] = val = val + buffer[tid + 256]; __syncthreads(); }
		if (CTA_SIZE_ >= 256) { if (tid < 128) buffer[tid] = val = val + buffer[tid + 128]; __syncthreads(); }
		if (CTA_SIZE_ >= 128) { if (tid <  64) buffer[tid] = val = val + buffer[tid + 64]; __syncthreads(); }

		if (tid < 32){
			if (CTA_SIZE_ >= 64) { buffer[tid] = val = val + buffer[tid + 32]; }
			if (CTA_SIZE_ >= 32) { buffer[tid] = val = val + buffer[tid + 16]; }
			if (CTA_SIZE_ >= 16) { buffer[tid] = val = val + buffer[tid + 8]; }
			if (CTA_SIZE_ >= 8) { buffer[tid] = val = val + buffer[tid + 4]; }
			if (CTA_SIZE_ >= 4) { buffer[tid] = val = val + buffer[tid + 2]; }
			if (CTA_SIZE_ >= 2) { buffer[tid] = val = val + buffer[tid + 1]; }
		}
	}

	__global__ void reduce_all_nodes_kernel(const float4* nodesDqVw, int n)
	{
		const float* beg = (const float*)nodesDqVw + blockIdx.x;
		float sum = 0.f;
		for (int i = threadIdx.x; i < n; i += blockDim.x)
			sum += beg[i * 12]; // dq+vw, 12 float per node

		__shared__ float smem[GpuGaussNewtonSolver::CTA_SIZE];

		smem[threadIdx.x] = sum;
		__syncthreads();

		reduce<GpuGaussNewtonSolver::CTA_SIZE>(smem);

		if (threadIdx.x == 0)
			_g_common_q[blockIdx.x] = smem[0];
	}


	__global__ void factor_all_nodes_kernel(float4* nodesDqVw, int n, Tbx::Dual_quat_cu rigid_inv)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n)
			return;
		
		Tbx::Dual_quat_cu dq = rigid_inv * pack_dual_quat(nodesDqVw[3 * i], nodesDqVw[3 * i + 1]);
		unpack_dual_quat(dq, nodesDqVw[3 * i], nodesDqVw[3 * i + 1]);
	}

	// optional, factor out common rigid transformations among all nodes
	void GpuGaussNewtonSolver::factor_out_rigid()
	{
		if (m_pWarpField == nullptr)
			throw std::exception("GpuGaussNewtonSolver::solve: null pointer");
		if (m_pWarpField->getNumLevels() < 2)
			throw std::exception("non-supported levels of warp field!");
		if (m_pWarpField->getNumNodesInLevel(0) == 0)
		{
			printf("no warp nodes, return\n");
			return;
		}
		const int num0 = m_pWarpField->getNumNodesInLevel(0);
		const int numAll = m_pWarpField->getNumAllNodes();

		Tbx::Dual_quat_cu dq(Tbx::Quat_cu(0,0,0,0), Tbx::Quat_cu(0,0,0,0));
		cudaMemcpyToSymbol(_g_common_q, &dq, sizeof(Tbx::Dual_quat_cu));

		reduce_all_nodes_kernel << <8, GpuGaussNewtonSolver::CTA_SIZE >> >(
			m_pWarpField->getNodesDqVwPtr(0), num0);
		cudaSafeCall(cudaGetLastError(), "reduce_all_nodes_kernel");
		cudaMemcpyFromSymbol(&dq, _g_common_q, sizeof(Tbx::Dual_quat_cu));

		if (dq.get_non_dual_part().norm() > Tbx::Dual_quat_cu::epsilon())
		{
			dq.normalize();
			m_pWarpField->set_rigidTransform(
				m_pWarpField->get_rigidTransform() * dq.to_transformation());

			for (int lv = 0; lv < m_pWarpField->getNumLevels(); lv++)
			{
				int numLv = m_pWarpField->getNumNodesInLevel(lv);
				if (numLv == 0)
					break;
				factor_all_nodes_kernel << <divUp(numLv, GpuGaussNewtonSolver::CTA_SIZE),
					GpuGaussNewtonSolver::CTA_SIZE >> >(m_pWarpField->getNodesDqVwPtr(lv), numLv, dq.conjugate());
			}
			cudaSafeCall(cudaGetLastError(), "factor_all_nodes_kernel");


			// re-extract info
			m_pWarpField->extract_nodes_info_no_allocation(m_nodesKnn, m_twist, m_nodesVw);
			checkNan(m_twist, numAll * 6, "twist after factoring rigid");
		}
	}
#pragma endregion

#pragma region --check linear solver
	void GpuGaussNewtonSolver::checkLinearSolver()
	{

	}
#pragma endregion
}