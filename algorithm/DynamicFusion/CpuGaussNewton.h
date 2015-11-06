#pragma once
#include "definations.h"
#include "DynamicFusionParam.h"
namespace dfusion
{
	class WarpField;
	struct EigenContainter;
	class CpuGaussNewton
	{
	public:
		CpuGaussNewton();
		~CpuGaussNewton();

		void init(WarpField* pWarpField, const MapArr& vmap_cano, const MapArr& nmap_cano,
			Param param, Intr intr);

		void findCorr(const MapArr& vmap_live, const MapArr& nmap_live, 
			const MapArr& vmap_warp, const MapArr& nmap_warp);

		void solve(bool factor_rigid_out = true);

		void debug_set_init_x(const float* x_host, int n);
	protected:
	private:
		WarpField* m_pWarpField;
		EigenContainter* m_egc;
		DeviceArray2D<KnnIdx> m_vmapKnn;
		DeviceArray<float> m_twist;
		DeviceArray<float4> m_vw;
		DeviceArray<KnnIdx> m_nodesKnn;
	};
}