#pragma once

#include "definations.h"
#include "DynamicFusionParam.h"
#include "WarpField.h"
namespace dfusion
{
	class GpuGaussNewtonSolver
	{
	public:
		enum
		{
			CTA_SIZE_X = 32,
			CTA_SIZE_Y = 8,
			CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,
			VarPerNode = 6,
			LowerPartNum = 21
		};
	public:
		GpuGaussNewtonSolver();
		~GpuGaussNewtonSolver();

		void init(WarpField* pWarpField, const MapArr& vmap_cano, 
			const MapArr& nmap_cano, Param param, Intr intr);

		void solve(const MapArr& vmap_live, const MapArr& nmap_live,
			const MapArr& vmap_warp, const MapArr& nmap_warp,
			bool factor_rigid_out = false);

		void debug_print();
		void debug_set_init_x(const float* x_host, int n);
	protected:
		void calcDataTerm();
		void bindTextures();
		void unBindTextures();
	private:
		WarpField* m_pWarpField;
		const MapArr* m_vmap_cano;
		const MapArr* m_nmap_cano;
		const MapArr* m_vmap_warp;
		const MapArr* m_nmap_warp;
		const MapArr* m_vmap_live;
		const MapArr* m_nmap_live;
		const Param* m_param;
		Intr m_intr;
		int m_numNodes;
		int m_numLv0Nodes;

		// for pre-allocation: allocate a lareger buffer than given nodes
		// to prevent allocation each frame
		int m_nodes_for_buffer;
		int m_not_lv0_nodes_for_buffer;

		DeviceArray2D<WarpField::KnnIdx> m_vmapKnn;
		DeviceArray<WarpField::KnnIdx> m_nodesKnn;
		DeviceArray<float4> m_nodesVw;

		// w.r.t x, the variables we will solve for.
		DeviceArray<float> m_twist;

		// the Hessian matrix representation:
		// H =	[ H_d B  ]
		//		[ B^t H_r]
		// note H is symmetric, thus it is enough to evaluate the lower triangular

		// Hessian of the left-top of data+reg term, 
		// it is a 6x6x? block diags.
		// It is symmetric, we only touch the lower part of each block
		// But for convinence, we allocate 6x6 buffer for each block
		DeviceArray<float> m_Hd;

		// Hessian of the bottom-right of the data+reg term, 
		// it is a dense matrix and symmetric
		// we allocate ?x? buffer but only touch the lower part.
		DeviceArray<float> m_Hr;

		// CSR sparse matrix
		DeviceArray<float> m_Bt_val;
		DeviceArray<int> m_Bt_RowPtr;
		DeviceArray<int> m_Bt_ColIdx;

		// m_g = -J^t * f
		// we will solve for H*m_h = m_g
		// and x += step * m_h
		DeviceArray<float> m_g;
		DeviceArray<float> m_h;
	};
}