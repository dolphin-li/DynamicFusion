#include "GpuGaussNewtonSolver.h"
namespace dfusion
{
#pragma comment(lib, "cusparse.lib")
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

		if (CUSPARSE_STATUS_SUCCESS != cusparseCreate(&m_cuSparseHandle))
			throw std::exception("cuSparse creating failed!");
	}

	GpuGaussNewtonSolver::~GpuGaussNewtonSolver()
	{
		unBindTextures();
		cusparseDestroy(m_cuSparseHandle);
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
		if (m_numNodes > USHRT_MAX)
			throw std::exception("not supported, too much nodes!");

		// sparse matrix info
		m_Jrrows = 0; // unknown now, decided later
		m_Jrcols = m_numNodes * VarPerNode;
		m_Jrnnzs = 0; // unknown now, decided later
		m_Brows = m_numLv0Nodes * VarPerNode;
		m_Bcols = notLv0Nodes * VarPerNode;
		m_Bnnzs = 0; // unknown now, decieded later
		m_HrRowsCols = m_Bcols;

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

			// each node create 2*3*k rows and each row create at most VarPerNode*2 cols
			m_Jr_RowCounter.create(m_nodes_for_buffer + 1);
			m_Jr_RowMap2NodeId.create(m_nodes_for_buffer * 6 * WarpField::KnnK + 1);
			m_Jr_RowPtr.create(m_nodes_for_buffer*6*WarpField::KnnK + 1);
			m_Jr_ColIdx.create(VarPerNode * 2  * m_Jr_RowPtr.size());
			m_Jr_val.create(m_Jr_ColIdx.size());
			m_Jr_RowPtr_coo.create(m_Jr_ColIdx.size());

			m_Jrt_RowPtr.create(m_nodes_for_buffer*VarPerNode + 1);
			m_Jrt_ColIdx.create(m_Jr_ColIdx.size());
			m_Jrt_val.create(m_Jr_ColIdx.size());
			m_Jrt_RowPtr_coo.create(m_Jr_ColIdx.size());

			// B = Jr0'Jr1
			m_B_RowPtr.create(m_nodes_for_buffer*VarPerNode + 1);
			m_B_ColIdx.create(WarpField::KnnK * VarPerNode * m_B_RowPtr.size());
			m_B_RowPtr_coo.create(m_B_ColIdx.size());
			m_B_val.create(m_B_ColIdx.size());

			m_Bt_RowPtr.create(m_nodes_for_buffer*VarPerNode + 1);
			m_Bt_ColIdx.create(m_B_ColIdx.size());
			m_Bt_RowPtr_coo.create(m_B_ColIdx.size());
			m_Bt_val.create(m_B_ColIdx.size());

			// the energy function of reg term
			m_f_r.create(m_Jr_RowPtr.size());
		}

		if (m_not_lv0_nodes_for_buffer < notLv0Nodes)
		{
			// the not-level0 nodes are not likely to increase dramatically
			// thus it is enough to allocate just a bit larger buffer
			m_not_lv0_nodes_for_buffer = notLv0Nodes * 1.2;
			m_Hr.create(m_not_lv0_nodes_for_buffer*m_not_lv0_nodes_for_buffer*
				VarPerNode * VarPerNode);
		}

		bindTextures();

		// extract knn map
		m_pWarpField->extract_knn_for_vmap(vmap_cano, m_vmapKnn);

		//extract nodes info
		m_pWarpField->extract_nodes_info_no_allocation(m_nodesKnn, m_twist, m_nodesVw);

		// the sparse block B
		initSparseStructure();
	}

	void GpuGaussNewtonSolver::solve(const MapArr& vmap_live, const MapArr& nmap_live,
		const MapArr& vmap_warp, const MapArr& nmap_warp,
		bool factor_rigid_out)
	{
		m_vmap_warp = &vmap_warp;
		m_nmap_warp = &nmap_warp;
		m_vmap_live = &vmap_live;
		m_nmap_live = &nmap_live;


		// perform Gauss-Newton iteration
		for (int iter = 0; iter < m_param->fusion_GaussNewton_maxIter; iter++)
		{
			cudaSafeCall(cudaMemset(m_Hd.ptr(), 0, sizeof(float)*m_Hd.size()));
			cudaSafeCall(cudaMemset(m_g.ptr(), 0, sizeof(float)*m_g.size()));

			// 1. calculate data term: Hd += Jd'Jd; g += Jd'fd
			calcDataTerm();

			// 2. calculate reg term: Jr = [Jr0 Jr1; 0 Jr3]; fr;
			calcRegTerm();

			// 3. calculate Hessian: Hd += Jr0'Jr0; B = Jr0'Jr1; Hr = Jr1'Jr1 + Jr3'Jr3; g+=Jr'*fr
			calcHessian();
		}// end for iter

		// finally, write results back
		m_pWarpField->update_nodes_via_twist(m_twist);
	}

	void GpuGaussNewtonSolver::debug_set_init_x(const float* x_host, int n)
	{
		if (n != m_numNodes*VarPerNode)
		{
			printf("debug_set_init_x: size not matched: %d %d\n", n, m_numNodes*VarPerNode);
			throw std::exception();
		}
		cudaSafeCall(cudaMemcpy(m_twist.ptr(), x_host, n*sizeof(float), cudaMemcpyHostToDevice));
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
					fprintf(pFile, "%f\n", host_g[i]);
			}
			fclose(pFile);
		}// dump the g

		dumpSparseMatrix("D:/tmp/gpu_Jr.txt", m_Jr_RowPtr, m_Jr_ColIdx, m_Jr_val, m_Jrrows);
		dumpSparseMatrix("D:/tmp/gpu_Jrt.txt", m_Jrt_RowPtr, m_Jrt_ColIdx, m_Jrt_val, m_Jrcols);
		dumpSparseMatrix("D:/tmp/gpu_B.txt", m_B_RowPtr, m_B_ColIdx, m_B_val, m_Brows);
		dumpSparseMatrix("D:/tmp/gpu_Bt.txt", m_Bt_RowPtr, m_Bt_ColIdx, m_Bt_val, m_Bcols);
		dumpSymLowerMat("D:/tmp/gpu_Hr.txt", m_Hr, m_HrRowsCols);
	}

	void GpuGaussNewtonSolver::dumpSparseMatrix(
		std::string name, const DeviceArray<int>& rptr,
		const DeviceArray<int>& cidx, const DeviceArray<float>& val, int nRow)
	{
		std::vector<int> hr, hc;
		std::vector<float> hv;

		rptr.download(hr);
		cidx.download(hc);
		val.download(hv);

		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int r = 0; r < nRow; r++)
			{
				int cb = hr[r], ce = hr[r + 1];
				for (int ic = cb; ic < ce; ic++)
					fprintf(pFile, "%d %d %f\n", r, hc[ic], hv[ic]);
				// if an empty row, fill diag with zero
				// this is for the convinience when exporting to matlab
				if (cb == ce)
					fprintf(pFile, "%d %d %f\n", r, r, 0);
			}
			fclose(pFile);
		}
	}

	void GpuGaussNewtonSolver::dumpSymLowerMat(std::string name, 
		const DeviceArray<float>& A, int nRowsCols)
	{
		std::vector<float> hA;
		A.download(hA);

		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < nRowsCols; y++)
			{
				for (int x = 0; x < nRowsCols; x++)
				{
					int x1 = x, y1 = y;
					if (x1 > y1)
						std::swap(x1, y1);
					fprintf(pFile, "%f ", hA[y1*nRowsCols + x1]);
				}
				fprintf(pFile, "\n");
			}
			fclose(pFile);
		}
	}
}