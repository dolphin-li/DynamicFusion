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
			const MapArr& vmap_warp, const MapArr& nmap_warp, Param param, Intr intr);

		void solve(const MapArr& vmap_live, const MapArr& nmap_live);
	protected:
	private:
		WarpField* m_pWarpField;
		EigenContainter* m_egc;
		DeviceArray2D<ushort4> m_vmapKnn;
		DeviceArray<float> m_twist;
		DeviceArray<float4> m_vw;
		DeviceArray<ushort4> m_nodesKnn;
	};
}