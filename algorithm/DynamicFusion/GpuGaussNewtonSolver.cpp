#include "GpuGaussNewtonSolver.h"
namespace dfusion
{
	GpuGaussNewtonSolver::GpuGaussNewtonSolver()
	{
		m_vmap_cano = nullptr;
		m_nmap_cano = nullptr;
		m_vmap_warp = nullptr;
		m_nmap_warp = nullptr;
		m_vmap_live = nullptr;
		m_nmap_live = nullptr;
		m_param = nullptr;
	}

	GpuGaussNewtonSolver::~GpuGaussNewtonSolver()
	{
	}

	void GpuGaussNewtonSolver::init(WarpField* pWarpField, const MapArr& vmap_cano, const MapArr& nmap_cano,
		Param param, Intr intr)
	{
		m_pWarpField = pWarpField;
		m_param = &param;
		m_intr = intr;

		m_vmap_cano = &vmap_cano;
		m_nmap_cano = &nmap_cano;

		// extract knn map
		m_pWarpField->extract_knn_for_vmap(vmap_cano, m_vmapKnn);

		//extract nodes info
		m_pWarpField->extract_nodes_info(m_nodesKnn, m_twist, m_vw); 
	}

	void GpuGaussNewtonSolver::solve(const MapArr& vmap_live, const MapArr& nmap_live,
		const MapArr& vmap_warp, const MapArr& nmap_warp,
		bool factor_rigid_out)
	{
		m_vmap_warp = &vmap_warp;
		m_nmap_warp = &nmap_warp;
		m_vmap_live = &vmap_live;
		m_nmap_live = &nmap_live;


		// finally, write results back
		m_pWarpField->update_nodes_via_twist(m_twist);
	}
}