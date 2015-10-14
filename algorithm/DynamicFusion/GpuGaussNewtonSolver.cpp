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
		m_nodes_for_buffer = 0;
		m_numNodes = 0;
		m_not_lv0_nodes_for_buffer = 0;
		m_numLv0Nodes = 0;
	}

	GpuGaussNewtonSolver::~GpuGaussNewtonSolver()
	{
	}

	void GpuGaussNewtonSolver::init(WarpField* pWarpField, const MapArr& vmap_cano, 
		const MapArr& nmap_cano, Param param, Intr intr)
	{
		m_pWarpField = pWarpField;
		m_param = &param;
		m_intr = intr;

		m_vmap_cano = &vmap_cano;
		m_nmap_cano = &nmap_cano;

		m_vmapKnn.create(vmap_cano.rows(), vmap_cano.cols());

		m_numNodes = pWarpField->getNumAllNodes();
		m_numLv0Nodes = pWarpField->getNumNodesInLevel(0);
		int notLv0Nodes = m_numNodes - m_numLv0Nodes;

		// make larger buffer to prevent malloc/free each frame
		if (m_nodes_for_buffer < m_numNodes)
		{
			// the nodes seems to increase frame-by-frame
			// thus we allocate enough big buffer to prevent re-allocation
			m_nodes_for_buffer = m_numNodes * 1.5;

			m_nodesKnn.create(m_nodes_for_buffer);
			m_twist.create(m_nodes_for_buffer * VarPerNode);
			m_nodesVw.create(m_nodes_for_buffer);
			m_h.create(m_nodes_for_buffer * VarPerNode);
			m_g.create(m_nodes_for_buffer * VarPerNode);
			m_Hd.create(VarPerNode * VarPerNode * m_nodes_for_buffer);
			m_Bt_ColIdx.create(VarPerNode*WarpField::KnnK*m_nodes_for_buffer);
			m_Bt_val.create(VarPerNode*WarpField::KnnK*m_nodes_for_buffer);
		}

		if (m_not_lv0_nodes_for_buffer < notLv0Nodes)
		{
			// the not-level0 nodes are not likely to increase dramatically
			// thus it is enough to allocate just a bit silghter buffer
			m_not_lv0_nodes_for_buffer = notLv0Nodes * 1.2;
			m_Hr.create(m_not_lv0_nodes_for_buffer*m_not_lv0_nodes_for_buffer*
				VarPerNode * VarPerNode);
			m_Bt_RowPtr.create(m_not_lv0_nodes_for_buffer);
		}

		// extract knn map
		m_pWarpField->extract_knn_for_vmap(vmap_cano, m_vmapKnn);

		//extract nodes info
		m_pWarpField->extract_nodes_info_no_allocation(m_nodesKnn, m_twist, m_nodesVw);
	}

	void GpuGaussNewtonSolver::solve(const MapArr& vmap_live, const MapArr& nmap_live,
		const MapArr& vmap_warp, const MapArr& nmap_warp,
		bool factor_rigid_out)
	{
		m_vmap_warp = &vmap_warp;
		m_nmap_warp = &nmap_warp;
		m_vmap_live = &vmap_live;
		m_nmap_live = &nmap_live;

		bindTextures();

		// perform Gauss-Newton iteration
		for (int iter = 0; iter < m_param->fusion_GaussNewton_maxIter; iter++)
		{
			// 1. calculate data term: H_d += J_dJ_d'; 
			cudaSafeCall(cudaMemset(m_Hd.ptr(), 0, sizeof(float)*m_Hd.size()));
			cudaSafeCall(cudaMemset(m_g.ptr(), 0, sizeof(float)*m_g.size()));
			calcDataTerm();
		}// end for iter

		unBindTextures();

		// finally, write results back
		m_pWarpField->update_nodes_via_twist(m_twist);
	}

	void GpuGaussNewtonSolver::debug_print()
	{
		// dump the diagonal block
		{
			std::vector<float> host_Hd;
			m_Hd.download(host_Hd);
			FILE* pFile = fopen("D:/tmp/gpu_Hd.txt", "w");
			if (pFile)
			{
				for (int i = 0; i < m_numLv0Nodes; i++)
				{
					const float* data = host_Hd.data() + i*VarPerNode*VarPerNode;
					for (int y = 0; y < VarPerNode; y++)
					for (int x = 0; x < VarPerNode; x++)
					{
						int x1 = x, y1 = y;
						if (x1 > y1)
							std::swap(x1, y1);
						float v = data[y1*VarPerNode+x1];
						fprintf(pFile, "%f ", v);
					}
					fprintf(pFile, "\n");
				}
			}
			fclose(pFile);
		}// dump the diaonal block

		// dump g
		{
			std::vector<float> host_g;
			m_g.download(host_g);
			FILE* pFile = fopen("D:/tmp/gpu_g.txt", "w");
			if (pFile)
			{
				for (int i = 0; i < m_numLv0Nodes*VarPerNode; i++)
				{
					fprintf(pFile, "%f\n", host_g[i]);
				}
			}
			fclose(pFile);
		}// dump the g
	}
}