#include "GpuGaussNewtonSolver.h"
namespace dfusion
{
#pragma comment(lib, "cusparse.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cusolver.lib")
	GpuGaussNewtonSolver::GpuGaussNewtonSolver()
	{
		if (CUSPARSE_STATUS_SUCCESS != cusparseCreate(&m_cuSparseHandle))
			throw std::exception("cuSparse creating failed!");

		if (CUBLAS_STATUS_SUCCESS != cublasCreate(&m_cublasHandle))
			throw std::exception("cuSparse creating failed!");

		if (CUSOLVER_STATUS_SUCCESS != cusolverDnCreate(&m_cuSolverHandle))
			throw std::exception("cusolverDnCreatefailed!");

		cusparseCreateMatDescr(&m_Jrt_desc);
		cusparseCreateMatDescr(&m_Bt_desc);
		cusparseCreateMatDescr(&m_B_desc);

		reset();
	}

	GpuGaussNewtonSolver::~GpuGaussNewtonSolver()
	{
		unBindTextures();
		cusolverDnDestroy(m_cuSolverHandle);
		cublasDestroy(m_cublasHandle);
		cusparseDestroyMatDescr(m_Jrt_desc);
		cusparseDestroyMatDescr(m_Bt_desc);
		cusparseDestroyMatDescr(m_B_desc);
		cusparseDestroy(m_cuSparseHandle);
	}

	template<class T>
	static void setzero(DeviceArray<T>& A)
	{
		cudaSafeCall(cudaMemset(A.ptr(), 0, A.sizeBytes()), "GpuGaussNewtonSolver::setZero");
	}

	void GpuGaussNewtonSolver::init(WarpField* pWarpField, const MapArr& vmap_cano, 
		const MapArr& nmap_cano, Param param, Intr intr)
	{
		if (pWarpField->getNumLevels() < 2)
			throw std::exception("non-supported levels of warp field!");

		bool nodesUpdated = false;

		m_pWarpField = pWarpField;
		m_param = &param;
		m_intr = intr;

		m_vmap_cano = &vmap_cano;
		m_nmap_cano = &nmap_cano;

		m_vmapKnn.create(vmap_cano.rows(), vmap_cano.cols());

		if (pWarpField->getNumNodesInLevel(0) != m_numLv0Nodes
			|| pWarpField->getNumAllNodes() != m_numNodes)
			nodesUpdated = true;
		m_numNodes = pWarpField->getNumAllNodes();
		m_numLv0Nodes = pWarpField->getNumNodesInLevel(0);
		int notLv0Nodes = m_numNodes - m_numLv0Nodes;
		if (m_numNodes > USHRT_MAX)
			throw std::exception("not supported, too much nodes!");

		// sparse matrix info
		if (nodesUpdated)
		{
			m_Jrrows = 0; // unknown now, decided later
			m_Jrcols = m_numNodes * VarPerNode;
			m_Jrnnzs = 0; // unknown now, decided later
			m_Brows = m_numLv0Nodes * VarPerNode;
			m_Bcols = notLv0Nodes * VarPerNode;
			m_Bnnzs = 0; // unknown now, decieded later
			m_HrRowsCols = m_Bcols;
		}

		// make larger buffer to prevent malloc/free each frame
		if (m_nodes_for_buffer < m_numNodes)
		{
			// the nodes seems to increase frame-by-frame
			// thus we allocate enough big buffer to prevent re-allocation
			m_nodes_for_buffer = m_numNodes * 1.5;

			m_nodesKnn.create(m_nodes_for_buffer);
			setzero(m_nodesKnn);
			m_twist.create(m_nodes_for_buffer * VarPerNode);
			setzero(m_twist);
			m_nodesVw.create(m_nodes_for_buffer);
			setzero(m_nodesVw);
			m_h.create(m_nodes_for_buffer * VarPerNode);
			setzero(m_h);
			m_g.create(m_nodes_for_buffer * VarPerNode);
			setzero(m_g);
			m_u.create(m_nodes_for_buffer * VarPerNode);
			setzero(m_u);
			m_tmpvec.create(m_nodes_for_buffer * VarPerNode);
			setzero(m_tmpvec);
			m_Hd.create(VarPerNode * VarPerNode * m_nodes_for_buffer);
			setzero(m_Hd);

			// each node create 2*3*k rows and each row create at most VarPerNode*2 cols
			m_Jr_RowCounter.create(m_nodes_for_buffer + 1);
			setzero(m_Jr_RowCounter);
			m_Jr_RowMap2NodeId.create(m_nodes_for_buffer * 6 * WarpField::KnnK + 1);
			setzero(m_Jr_RowMap2NodeId);
			m_Jr_RowPtr.create(m_nodes_for_buffer * 6 * WarpField::KnnK + 1);
			setzero(m_Jr_RowPtr);
			m_Jr_ColIdx.create(VarPerNode * 2 * m_Jr_RowPtr.size());
			setzero(m_Jr_ColIdx);
			m_Jr_val.create(m_Jr_ColIdx.size());
			setzero(m_Jr_val);
			m_Jr_RowPtr_coo.create(m_Jr_ColIdx.size());
			setzero(m_Jr_RowPtr_coo);

			m_Jrt_RowPtr.create(m_nodes_for_buffer*VarPerNode + 1);
			setzero(m_Jrt_RowPtr);
			m_Jrt_ColIdx.create(m_Jr_ColIdx.size());
			setzero(m_Jrt_ColIdx);
			m_Jrt_val.create(m_Jr_ColIdx.size());
			setzero(m_Jrt_val);
			m_Jrt_RowPtr_coo.create(m_Jr_ColIdx.size());
			setzero(m_Jrt_RowPtr_coo);

			// B = Jr0'Jr1
			m_B_RowPtr.create(m_nodes_for_buffer*VarPerNode + 1);
			setzero(m_B_RowPtr);
			m_B_ColIdx.create(WarpField::KnnK * VarPerNode * m_B_RowPtr.size());
			setzero(m_B_ColIdx);
			m_B_RowPtr_coo.create(m_B_ColIdx.size());
			setzero(m_B_RowPtr_coo);
			m_B_val.create(m_B_ColIdx.size());
			setzero(m_B_val);

			m_Bt_RowPtr.create(m_nodes_for_buffer*VarPerNode + 1);
			setzero(m_Bt_RowPtr);
			m_Bt_ColIdx.create(m_B_ColIdx.size());
			setzero(m_Bt_ColIdx);
			m_Bt_RowPtr_coo.create(m_B_ColIdx.size());
			setzero(m_Bt_RowPtr_coo);
			m_Bt_val.create(m_B_ColIdx.size());
			setzero(m_Bt_val);
			m_Bt_Ltinv_val.create(m_B_ColIdx.size());
			setzero(m_Bt_Ltinv_val);

			// the energy function of reg term
			m_f_r.create(m_Jr_RowPtr.size());
			setzero(m_f_r);

			// for block solver
			m_Hd_Linv.create(m_Hd.size());
			setzero(m_Hd_Linv);
			m_Hd_LLtinv.create(m_Hd.size());
			setzero(m_Hd_LLtinv);
		}

		if (m_not_lv0_nodes_for_buffer < notLv0Nodes)
		{
			// the not-level0 nodes are not likely to increase dramatically
			// thus it is enough to allocate just a bit larger buffer
			m_not_lv0_nodes_for_buffer = notLv0Nodes * 1.2;
			m_Hr.create(m_not_lv0_nodes_for_buffer*m_not_lv0_nodes_for_buffer*
				VarPerNode * VarPerNode);
			setzero(m_Hr);
			m_Q.create(m_Hr.size());
			setzero(m_Q);
		}

		bindTextures();

		// extract knn map
		m_pWarpField->extract_knn_for_vmap(vmap_cano, m_vmapKnn);

		//extract nodes info
		m_pWarpField->extract_nodes_info_no_allocation(m_nodesKnn, m_twist, m_nodesVw);

		// the sparse block B
		if (nodesUpdated)
			initSparseStructure();
	}

	void GpuGaussNewtonSolver::reset()
	{
		m_vmap_cano = nullptr;
		m_nmap_cano = nullptr;
		m_vmap_warp = nullptr;
		m_nmap_warp = nullptr;
		m_vmap_live = nullptr;
		m_nmap_live = nullptr;
		m_param = nullptr;
		m_numNodes = 0;
		m_numLv0Nodes = 0;

		m_Jrrows = 0;
		m_Jrcols = 0;
		m_Jrnnzs = 0;
		m_Brows = 0;
		m_Bcols = 0;
		m_Bnnzs = 0;
		m_HrRowsCols = 0;

		setzero(m_nodesKnn);
		setzero(m_twist);
		setzero(m_nodesVw);
		setzero(m_h);
		setzero(m_g);
		setzero(m_u);
		setzero(m_tmpvec);
		setzero(m_Hd);

		// each node create 2*3*k rows and each row create at most VarPerNode*2 cols
		setzero(m_Jr_RowCounter);
		setzero(m_Jr_RowMap2NodeId);
		setzero(m_Jr_RowPtr);
		setzero(m_Jr_ColIdx);
		setzero(m_Jr_val);
		setzero(m_Jr_RowPtr_coo);

		setzero(m_Jrt_RowPtr);
		setzero(m_Jrt_ColIdx);
		setzero(m_Jrt_val);
		setzero(m_Jrt_RowPtr_coo);

		// B = Jr0'Jr1
		setzero(m_B_RowPtr);
		setzero(m_B_ColIdx);
		setzero(m_B_RowPtr_coo);
		setzero(m_B_val);

		setzero(m_Bt_RowPtr);
		setzero(m_Bt_ColIdx);
		setzero(m_Bt_RowPtr_coo);
		setzero(m_Bt_val);
		setzero(m_Bt_Ltinv_val);

		// the energy function of reg term
		setzero(m_f_r);

		// for block solver
		setzero(m_Hd_Linv);
		setzero(m_Hd_LLtinv);

		setzero(m_Hr);
		setzero(m_Q);
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
		//for (int k = 0; k < 100; k++)
		for (int iter = 0; iter < m_param->fusion_GaussNewton_maxIter; iter++)
		{
			cudaSafeCall(cudaMemset(m_Hd.ptr(), 0, sizeof(float)*m_Hd.size()));
			cudaSafeCall(cudaMemset(m_g.ptr(), 0, sizeof(float)*m_g.size()));

			// 1. calculate data term: Hd += Jd'Jd; g += Jd'fd
			calcDataTerm();

			// 2. calculate reg term: Jr = [Jr0 Jr1; 0 Jr3]; fr;
			calcRegTerm();

			// 3. calculate Hessian: Hd += Jr0'Jr0; B = Jr0'Jr1; Hr = Jr1'Jr1 + Jr3'Jr3; g=-(g+Jr'*fr)
			calcHessian();

			// 4. solve H*h = g
			blockSolve();

			// 5. accumulate: x += step * h;
			cublasSaxpy(m_cublasHandle, m_Jrcols, &m_param->fusion_GaussNewton_fixedStep, 
				m_h.ptr(), 1, m_twist.ptr(), 1);
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
		dumpBlocks("D:/tmp/gpu_Hd.txt", m_Hd, m_numLv0Nodes, VarPerNode);
		dumpBlocks("D:/tmp/gpu_Hd_Linv.txt", m_Hd_Linv, m_numLv0Nodes, VarPerNode);
		dumpBlocks("D:/tmp/gpu_Hd_LLtinv.txt", m_Hd_LLtinv, m_numLv0Nodes, VarPerNode);
		dumpVec("D:/tmp/gpu_g.txt", m_g, m_numNodes*VarPerNode);
		dumpSparseMatrix("D:/tmp/gpu_Jr.txt", m_Jr_RowPtr, m_Jr_ColIdx, m_Jr_val, m_Jrrows);
		dumpSparseMatrix("D:/tmp/gpu_Jrt.txt", m_Jrt_RowPtr, m_Jrt_ColIdx, m_Jrt_val, m_Jrcols);
		dumpSparseMatrix("D:/tmp/gpu_B.txt", m_B_RowPtr, m_B_ColIdx, m_B_val, m_Brows);
		dumpSparseMatrix("D:/tmp/gpu_Bt.txt", m_Bt_RowPtr, m_Bt_ColIdx, m_Bt_val, m_Bcols);
		dumpSparseMatrix("D:/tmp/gpu_BtLtinv.txt", m_Bt_RowPtr, m_Bt_ColIdx, m_Bt_Ltinv_val, m_Bcols);
		dumpMat("D:/tmp/gpu_Hr.txt", m_Hr, m_HrRowsCols);
		dumpMat("D:/tmp/gpu_Q.txt", m_Q, m_HrRowsCols);
		dumpVec("D:/tmp/gpu_fr.txt", m_f_r, m_Jrrows);
		dumpVec("D:/tmp/gpu_g.txt", m_g, m_Jrcols);
		dumpVec("D:/tmp/gpu_u.txt", m_u, m_Jrcols);
		dumpVec("D:/tmp/gpu_h.txt", m_h, m_Jrcols);
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
					fprintf(pFile, "%d %d %ef\n", r, hc[ic], hv[ic]);
				// if an empty row, fill diag with zero
				// this is for the convinience when exporting to matlab
				if (cb == ce)
					fprintf(pFile, "%d %d %ef\n", r, r, 0);
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
					fprintf(pFile, "%ef ", hA[y1*nRowsCols + x1]);
				}
				fprintf(pFile, "\n");
			}
			fclose(pFile);
		}
	}

	void GpuGaussNewtonSolver::dumpMat(std::string name,
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
					fprintf(pFile, "%ef ", hA[y*nRowsCols + x]);
				fprintf(pFile, "\n");
			}
			fclose(pFile);
		}
	}

	void GpuGaussNewtonSolver::dumpVec(std::string name, const DeviceArray<float>& A, int n)
	{
		std::vector<float> hA;
		A.download(hA);

		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int y = 0; y < n; y++)
				fprintf(pFile, "%ef\n", hA[y]);
			fclose(pFile);
		}
	}

	void GpuGaussNewtonSolver::dumpBlocks(std::string name, const DeviceArray<float>& A, 
		int nBlocks, int blockRowCol)
	{
		std::vector<float> hA;
		A.download(hA);

		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			int sz = blockRowCol*blockRowCol;
			for (int i = 0; i < nBlocks; i++)
			{
				for (int x = 0; x < sz; x++)
					fprintf(pFile, "%ef ", hA[i*sz + x]);
				fprintf(pFile, "\n");
			}
			fclose(pFile);
		}
	}
}