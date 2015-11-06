#include "GpuGaussNewtonSolver.h"
#include <string>
namespace dfusion
{
#define ENABLE_NAN_CHECKING
#pragma comment(lib, "cusparse.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cusolver.lib")
#define CHECK(a, msg){if(!(a)) throw std::exception(msg);} 
#define CHECK_LE(a, b){if((a) > (b)) {std::cout << "" << #a << "(" << a << ")<=" << #b << "(" << b << ")";throw std::exception(" ###error!");}} 

	GpuGaussNewtonSolver::GpuGaussNewtonSolver()
	{
		if (CUSPARSE_STATUS_SUCCESS != cusparseCreate(&m_cuSparseHandle))
			throw std::exception("cuSparse creating failed!");
		cusparseSetPointerMode(m_cuSparseHandle, CUSPARSE_POINTER_MODE_HOST);

		if (CUBLAS_STATUS_SUCCESS != cublasCreate(&m_cublasHandle))
			throw std::exception("cuSparse creating failed!");

		if (CUSOLVER_STATUS_SUCCESS != cusolverDnCreate(&m_cuSolverHandle))
			throw std::exception("cusolverDnCreatefailed!");

		m_nodes_for_buffer = 0;
		m_pWarpField = nullptr;
		m_Jr = new CudaBsrMatrix(m_cuSparseHandle);
		m_Jrt = new CudaBsrMatrix(m_cuSparseHandle);
		m_Jrt13_structure = new CudaBsrMatrix(m_cuSparseHandle, true);
		m_Jr13_structure = new CudaBsrMatrix(m_cuSparseHandle, true);
		m_B = new CudaBsrMatrix(m_cuSparseHandle);
		m_Bt = new CudaBsrMatrix(m_cuSparseHandle);
		m_Bt_Ltinv = new CudaBsrMatrix(m_cuSparseHandle);
		m_Hr = new CudaBsrMatrix(m_cuSparseHandle);
		m_H_singleLevel = new CudaBsrMatrix(m_cuSparseHandle);
		m_Q = new CudaBsrMatrix(m_cuSparseHandle);
		m_singleLevel_solver = new CudaCholeskeySolver();
		m_singleLevel_solver->init();

		reset();
	}

	GpuGaussNewtonSolver::~GpuGaussNewtonSolver()
	{
		unBindTextures();
		cusolverDnDestroy(m_cuSolverHandle);
		cublasDestroy(m_cublasHandle);
		cusparseDestroy(m_cuSparseHandle);
		delete m_Jr;
		delete m_Jrt;
		delete m_Jrt13_structure;
		delete m_Jr13_structure;
		delete m_B;
		delete m_Bt;
		delete m_Bt_Ltinv;
		delete m_Hr;
		delete m_H_singleLevel;
		delete m_singleLevel_solver;
		delete m_Q;
	}

	template<class T>
	static void setzero(DeviceArray<T>& A)
	{
		cudaSafeCall(cudaMemset(A.ptr(), 0, A.sizeBytes()), "GpuGaussNewtonSolver::setZero");
	}

	void GpuGaussNewtonSolver::checkNan(const DeviceArray<float>& x, int n, const char* msg)
	{
#ifdef ENABLE_NAN_CHECKING
		if (m_param->solver_enable_nan_check)
		{
			static int a = 0;
			std::vector<float> hx;
			x.download(hx);
			for (size_t i = 0; i < n; i++)
			{
				if (isnan(hx.at(i)))
				{
					printf("[%d]nan(%s): %d %f\n", a, msg, i, hx[i]);
					debug_print();
					system("pause");
				}
				if (isinf(hx.at(i)))
				{
					printf("[%d]inf(%s): %d %f\n", a, msg, i, hx[i]);
					debug_print();
					system("pause");
				}
			}
			a++;
		}
#endif
	}

	void GpuGaussNewtonSolver::init(WarpField* pWarpField, const MapArr& vmap_cano, 
		const MapArr& nmap_cano, Param param, Intr intr)
	{
		if (pWarpField->getNumNodesInLevel(0) == 0)
		{
			printf("no warp nodes, return\n");
			return;
		}

		bool nodesUpdated = false;

		m_pWarpField = pWarpField;
		m_param = &param;
		m_intr = intr;

		m_vmap_cano = &vmap_cano;
		m_nmap_cano = &nmap_cano;

		if (vmap_cano.rows() != m_vmapKnn.rows() || vmap_cano.cols() != m_vmapKnn.cols())
		{
			m_vmapKnn.create(vmap_cano.rows(), vmap_cano.cols());
			cudaMemset2D(m_vmapKnn.ptr(), m_vmapKnn.step(), 0, m_vmapKnn.cols(), m_vmapKnn.rows());
		}

		if (pWarpField->getNumNodesInLevel(0) != m_numLv0Nodes
			|| pWarpField->getNumAllNodes() != m_numNodes)
			nodesUpdated = true;
		m_numNodes = pWarpField->getNumAllNodes();
		m_numLv0Nodes = pWarpField->getNumNodesInLevel(0);
		int notLv0Nodes = m_numNodes - m_numLv0Nodes;
		if (m_numNodes > USHRT_MAX)
			throw std::exception("not supported, too much nodes!");

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

			// each node create 2*3*k rows and each row create at most VarPerNode*2 cols
			m_Jr_RowCounter.create(m_nodes_for_buffer + 1);
			setzero(m_Jr_RowCounter);
			m_Jr_RowMap2NodeId.create(m_nodes_for_buffer * 6 * WarpField::KnnK + 1);
			setzero(m_Jr_RowMap2NodeId);

			// the energy function of reg term
			m_f_r.create(m_nodes_for_buffer*VarPerNode*6 + 1);
			setzero(m_f_r);

			// for total energy evaluation
			m_energy_vec.create(m_vmapKnn.rows()*m_vmapKnn.cols() + m_nodes_for_buffer*WarpField::KnnK);
		}

		bindTextures();

		// extract knn map
		m_pWarpField->extract_knn_for_vmap(vmap_cano, m_vmapKnn);

		//extract nodes info
		m_pWarpField->extract_nodes_info_no_allocation(m_nodesKnn, m_twist, m_nodesVw);
		checkNan(m_twist, m_numNodes, "twist_init");

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

		cudaMemset2D(m_vmapKnn.ptr(), m_vmapKnn.step(), 0, 
			m_vmapKnn.cols(), m_vmapKnn.rows());

		setzero(m_nodesKnn);
		setzero(m_twist);
		setzero(m_nodesVw);
		setzero(m_h);
		setzero(m_g);
		setzero(m_u);
		setzero(m_tmpvec);
		m_Hd = 0.f;

		// each node create 2*3*k rows and each row create at most VarPerNode*2 cols
		setzero(m_Jr_RowCounter);
		setzero(m_Jr_RowMap2NodeId);
		*m_Jr = 0.f;
		*m_Jrt = 0.f;

		// B = Jr0'Jr1
		*m_B = 0.f;
		*m_Bt = 0.f;
		*m_Bt_Ltinv = 0.f;

		// the energy function of reg term
		setzero(m_f_r);

		// for block solver
		m_Hd_Linv = 0.f;
		m_Hd_LLtinv = 0.f;

		*m_Hr = 0.f;
		*m_Q = 0.f;

		*m_H_singleLevel = 0.f;
	}

	float GpuGaussNewtonSolver::solve(const MapArr& vmap_live, const MapArr& nmap_live,
		const MapArr& vmap_warp, const MapArr& nmap_warp, float* data_energy_, float* reg_energy_)
	{
		if (m_pWarpField == nullptr)
			throw std::exception("GpuGaussNewtonSolver::solve: null pointer");
		if (m_pWarpField->getNumNodesInLevel(0) == 0)
		{
			printf("no warp nodes, return\n");
			return FLT_MAX;
		}

		m_vmap_warp = &vmap_warp;
		m_nmap_warp = &nmap_warp;
		m_vmap_live = &vmap_live;
		m_nmap_live = &nmap_live;

		// perform Gauss-Newton iteration
		//for (int k = 0; k < 100; k++)
		float totalEnergy = 0.f, data_energy=0.f, reg_energy=0.f;
		m_pWarpField->extract_nodes_info_no_allocation(m_nodesKnn, m_twist, m_nodesVw);
		for (int iter = 0; iter < m_param->fusion_GaussNewton_maxIter; iter++)
		{
			m_Hd = 0.f;
			cudaSafeCall(cudaMemset(m_g.ptr(), 0, sizeof(float)*m_g.size()), 
				"GpuGaussNewtonSolver::solve, setg=0");

			checkNan(m_twist, m_numNodes, ("twist_" + std::to_string(iter)).c_str());

			// 1. calculate data term: Hd += Jd'Jd; g += Jd'fd
			calcDataTerm();
			checkNan(m_Hd.toDeviceArray(), m_numLv0Nodes*VarPerNode*VarPerNode, 
				("Hd_data_" + std::to_string(iter)).c_str());
			checkNan(m_g, m_numLv0Nodes*VarPerNode, ("g_data_" + std::to_string(iter)).c_str());
			
			// 2. calculate reg term: Jr = [Jr0 Jr1; 0 Jr3]; fr;
			calcRegTerm();
			checkNan(m_f_r, m_Jr->rows(), ("fr_" + std::to_string(iter)).c_str());

			// 3. calculate Hessian: Hd += Jr0'Jr0; B = Jr0'Jr1; Hr = Jr1'Jr1 + Jr3'Jr3; g=-(g+Jr'*fr)
			calcHessian();
			checkNan(m_Hd.toDeviceArray(), m_numLv0Nodes*VarPerNode*VarPerNode, 
				("Hd_reg_" + std::to_string(iter)).c_str());
			checkNan(m_g, m_Jr->cols(), ("g_" + std::to_string(iter)).c_str());

			// 4. solve H*h = g
			if (m_param->graph_single_level)
				singleLevelSolve();
			else
				blockSolve();
			checkNan(m_h, m_Jr->cols(), ("h_" + std::to_string(iter)).c_str());

			//debug_print();
			//printf("@\n");
			//system("pause");

			// if not fix step, we perform line search
			if (m_param->fusion_GaussNewton_fixedStep <= 0.f)
			{
				float old_energy = calcTotalEnergy(data_energy, reg_energy);
				float new_energy = 0.f;
				float alpha = 1.f;
				const static float alpha_stop = 1e-2;
				cudaSafeCall(cudaMemcpy(m_tmpvec.ptr(), m_twist.ptr(), m_Jr->cols()*sizeof(float),
					cudaMemcpyDeviceToDevice), "copy tmp vec to twist");
				for (; alpha > alpha_stop; alpha *= 0.5)
				{
					// x += alpha * h
					updateTwist_inch(m_h.ptr(), alpha);
					new_energy = calcTotalEnergy(data_energy, reg_energy);
					if (new_energy < old_energy)
						break;
					// reset x
					cudaSafeCall(cudaMemcpy(m_twist.ptr(), m_tmpvec.ptr(), 
					m_Jr->cols()*sizeof(float), cudaMemcpyDeviceToDevice), "copy twist to tmp vec");
				}
				totalEnergy = new_energy;
				if (alpha <= alpha_stop)
					break;
				float norm_h = 0.f, norm_g = 0.f;
				cublasStatus_t st = cublasSnrm2(m_cublasHandle, m_Jr->cols(),
					m_h.ptr(), 1, &norm_h);
				st = cublasSnrm2(m_cublasHandle, m_Jr->cols(),
					m_g.ptr(), 1, &norm_g);
				if (norm_h < (norm_g + 1e-6f) * 1e-6f)
					break;
			}
			// else, we perform fixed step update.
			else
			{
				// 5. accumulate: x += step * h;
				updateTwist_inch(m_h.ptr(), m_param->fusion_GaussNewton_fixedStep);
			}
		}// end for iter

		if (m_param->fusion_GaussNewton_fixedStep > 0.f)
			totalEnergy = calcTotalEnergy(data_energy, reg_energy);

		if (data_energy_)
			*data_energy_ = data_energy;
		if (reg_energy_)
			*reg_energy_ = reg_energy;

		return totalEnergy;
	}

	// update warpField by calling this function explicitly
	void GpuGaussNewtonSolver::updateWarpField()
	{
		if (m_pWarpField == nullptr)
			throw std::exception("GpuGaussNewtonSolver::solve: null pointer");
		if (m_pWarpField->getNumNodesInLevel(0) == 0)
		{
			printf("no warp nodes, return\n");
			return;
		}
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
		cudaSafeCall(cudaMemcpy(m_twist.ptr(), x_host, n*sizeof(float), cudaMemcpyHostToDevice),
			"GpuGaussNewtonSolver::debug_set_init_x");
	}

	void GpuGaussNewtonSolver::debug_print()
	{
		dumpBlocks("D:/tmp/gpu_Hd.txt", m_Hd.toDeviceArray(), m_numLv0Nodes, VarPerNode);
		dumpBlocks("D:/tmp/gpu_Hd_Linv.txt", m_Hd_Linv.toDeviceArray(), m_numLv0Nodes, VarPerNode);
		dumpBlocks("D:/tmp/gpu_Hd_LLtinv.txt", m_Hd_LLtinv.toDeviceArray(), m_numLv0Nodes, VarPerNode);
		dumpVec("D:/tmp/gpu_g.txt", m_g, m_numNodes*VarPerNode);
		m_Jr->dump("D:/tmp/gpu_Jr.txt");
		m_Jrt->dump("D:/tmp/gpu_Jrt.txt");
		m_B->dump("D:/tmp/gpu_B.txt");
		m_Bt->dump("D:/tmp/gpu_Bt.txt");
		m_Bt_Ltinv->dump("D:/tmp/gpu_BtLtinv.txt");
		m_Hr->dump("D:/tmp/gpu_Hr.txt");
		m_H_singleLevel->dump("D:/tmp/gpu_H_singleLevl.txt");
		m_Q->dump("D:/tmp/gpu_Q.txt");
		dumpVec("D:/tmp/gpu_fr.txt", m_f_r, m_Jr->rows());
		dumpVec("D:/tmp/gpu_g.txt", m_g, m_Jr->cols());
		dumpVec("D:/tmp/gpu_u.txt", m_u, m_Jr->cols());
		dumpVec("D:/tmp/gpu_h.txt", m_h, m_Jr->cols());
		dumpVec("D:/tmp/gpu_twist.txt", m_twist, m_Jr->cols());
		dumpQuats("D:/tmp/gpu_quats.txt", m_twist, m_Jr->cols());
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

	void GpuGaussNewtonSolver::dumpQuats(std::string name, const DeviceArray<float>& twist, int n)
	{
		int nNodes = n / 6;
		if (nNodes * 6 != n)
			throw std::exception("error: twist size incorrect");

		std::vector<float> hA;
		twist.download(hA);

		FILE* pFile = fopen(name.c_str(), "w");
		if (pFile)
		{
			for (int i = 0; i < n; i+=6)
			{
				Tbx::Vec3 r(hA[i + 0], hA[i + 1], hA[i + 2]);
				Tbx::Vec3 t(hA[i + 3], hA[i + 4], hA[i + 5]);
				Tbx::Dual_quat_cu dq;
				dq.from_twist(r, t);
				fprintf(pFile, "%ef %ef %ef; %ef %ef %ef; %ef %ef %ef %ef; %ef %ef %ef %ef\n", 
					r[0], r[1], r[2], t[0], t[1], t[2],
					dq[0], dq[1], dq[2], dq[3], dq[4], dq[5], dq[6], dq[7]
					);
			}
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

	void GpuGaussNewtonSolver::calcHessian()
	{
		if (m_param->graph_single_level)
		{
			// 1. compute H = Jr'*Jr + Hd
			// 1.1 Previously, the lower part of Hd is calculated, now we complete it
			m_Hd.transpose_L_to_U();
			m_Jrt->multBsrT_addDiag_value(*m_Jrt, *m_H_singleLevel,
				1.f, &m_Hd, 1.f);
			m_H_singleLevel->axpy_diag(1.f + m_param->fusion_GaussNewton_diag_regTerm);

			// 2. compute g = -(g + Jr'*fr)
			m_Jrt->Mv(m_f_r, m_g, -1.f, -1.f);
		}
		else
		{
			// 1. compute Jr0'Jr0 and accumulate into Hd
			m_Jrt->range(0, 0, m_B->blocksInRow(), m_Jrt->blocksInCol()).AAt_blockDiags(
				m_Hd, true, 1, 1);
			m_Hd.axpy_diag(1 + m_param->fusion_GaussNewton_diag_regTerm);

			// 1.1 fill the upper tri part of Hd
			// previously, we only calculate the lower triangular pert of Hd;
			// now that the computation of Hd is ready, we fill the mission upper part
			m_Hd.transpose_L_to_U();

			// 2. compute B = Jr0'Jr1
			m_Jrt->range(0, 0, m_B->blocksInRow(), m_Jrt->blocksInCol()).multBsrT_value(
				m_Jrt->range(m_B->blocksInRow(), 0, m_Jrt->blocksInRow(), m_Jrt->blocksInCol()), *m_B);

			// 3. compute Bt
			m_B->transposeValueTo(*m_Bt);

			// 4. compute Hr
			m_Jrt->range(m_numLv0Nodes, 0, m_Jrt->blocksInRow(), m_Jrt->blocksInCol()).multBsrT_value(
				m_Jrt->range(m_numLv0Nodes, 0, m_Jrt->blocksInRow(), m_Jrt->blocksInCol()), *m_Hr);
			m_Hr->axpy_diag(1 + m_param->fusion_GaussNewton_diag_regTerm);

			// 5. compute g = -(g + Jr'*fr)
			m_Jrt->Mv(m_f_r, m_g, -1.f, -1.f);
		}
	}

	void GpuGaussNewtonSolver::singleLevelSolve()
	{
		m_singleLevel_solver->factor();
		m_singleLevel_solver->solve(m_h, m_g);

		//checkLinearSolver(m_H_singleLevel, m_h, m_g);
	}

	void GpuGaussNewtonSolver::blockSolve()
	{
		// 1. batch LLt the diag blocks Hd==================================================
		m_Hd_Linv = m_Hd;
		m_Hd_Linv.cholesky().invL();
		m_Hd_Linv.LtL(m_Hd_LLtinv);

		// 2. compute Q = Hr - Bt * inv(Hd) * B ======================================
		// 2.1 compute Bt*Ltinv
		m_Bt->rightMultDiag_value(m_Hd_Linv, *m_Bt_Ltinv, true, true);

		// 2.2 compute Q
		m_Bt_Ltinv->multBsrT_value(*m_Bt_Ltinv, *m_Q, -1.f, m_Hr, 1.f);
		m_singleLevel_solver->factor();

		// 4. solve H*h = g =============================================================
		const int sz = m_Jr->cols();
		const int sz0 = m_B->rows();
		const int sz1 = sz - sz0;
		CHECK_LE(sz, m_u.size());
		CHECK_LE(sz, m_h.size());
		CHECK_LE(sz, m_g.size());
		CHECK_LE(sz, m_tmpvec.size());

		// 4.1 let H = LL', first we solve for L*u=g;
		// 4.1.1 u(0:sz0-1) = HdLinv*g(0:sz0-1)
		m_Hd_Linv.Lv(m_g.ptr(), m_u.ptr());

		// 4.1.2 u(sz0:sz-1) = LQinv*(g(sz0:sz-1) - Bt*HdLtinv*HdLinv*g(0:sz0-1))
		if (sz1 > 0)
		{
			// tmpvec = HdLtinv*HdLinv*g(0:sz0-1)
			m_Hd_Linv.Ltv(m_u.ptr(), m_tmpvec.ptr());

			// u(sz0:sz-1) = g(sz0:sz-1) - Bt*tmpvec
			cudaMemcpy(m_u.ptr() + sz0, m_g.ptr() + sz0, sz1*sizeof(float), cudaMemcpyDeviceToDevice);
			m_Bt->Mv(m_tmpvec.ptr(), m_u.ptr() + sz0, -1.f, 1.f);

			// solve LQ*u(sz0:sz-1) = u(sz0:sz-1)
			m_singleLevel_solver->solveL(m_u.ptr() + sz0, m_u.ptr() + sz0);
		}
		checkNan(m_u, sz, "u");

		// 4.2 then we solve for L'*h=u;
		// 4.2.1 h(sz0:sz-1) = UQinv*u(sz0:sz-1)
		if (sz1 > 0)
			m_singleLevel_solver->solveLt(m_h.ptr() + sz0, m_u.ptr() + sz0);

		// 4.2.2 h(0:sz0-1) = HdLtinv*( u(0:sz0-1) - HdLinv*B*h(sz0:sz-1) )
		// tmpvec = B*h(sz0:sz-1)
		m_B->Mv(m_h.ptr() + sz0, m_tmpvec.ptr());

		// u(0:sz0-1) = u(0:sz0-1) - HdLinv * tmpvec
		// h(0:sz0-1) = HdLtinv*u(0:sz0-1)
		if (sz1 > 0)
		{
			m_Hd_Linv.Lv(m_tmpvec.ptr(), m_u.ptr(), -1.f, 1.f);
			m_Hd_Linv.Ltv(m_u.ptr(), m_h.ptr());
		}
		else
			m_Hd_Linv.Ltv(m_u.ptr(), m_h.ptr(), -1.f);
	}

	void GpuGaussNewtonSolver::checkLinearSolver(const CudaBsrMatrix* A, const float* x, const float* b)
	{
		if (m_tmpvec.size() < A->rows())
			m_tmpvec.create(A->rows());

		cudaSafeCall(cudaMemcpy(m_tmpvec, b, A->rows()*sizeof(float), 
			cudaMemcpyDeviceToDevice), "GpuGaussNewtonSolver::checkLinearSolver, 1");

		// tmp = A*x-b
		A->Mv(x, m_tmpvec, 1.f, -1.f);

		float dif = 0.f, nb;
		cublasSnrm2(m_cublasHandle, A->rows(), m_tmpvec, 1, &dif);
		cublasSnrm2(m_cublasHandle, A->rows(), b, 1, &nb);

		dif /= nb;
		printf("linear solver, dif: %ef\n", dif);
	}
}