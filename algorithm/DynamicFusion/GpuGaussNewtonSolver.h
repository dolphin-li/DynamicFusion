#pragma once

#include "definations.h"
#include "DynamicFusionParam.h"
#include "WarpField.h"
namespace dfusion
{
	class GpuGaussNewtonSolver
	{
	public:
		GpuGaussNewtonSolver();
		~GpuGaussNewtonSolver();

		void init(WarpField* pWarpField, const MapArr& vmap_cano, 
			const MapArr& nmap_cano, Param param, Intr intr);

		void solve(const MapArr& vmap_live, const MapArr& nmap_live,
			const MapArr& vmap_warp, const MapArr& nmap_warp,
			bool factor_rigid_out = false);
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

		DeviceArray2D<WarpField::KnnIdx> m_vmapKnn;
		DeviceArray<WarpField::KnnIdx> m_nodesKnn;
		DeviceArray<float> m_twist;
		DeviceArray<float4> m_vw;
	};
}