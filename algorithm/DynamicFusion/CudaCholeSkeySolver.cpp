#include "CudaCholeskeySolver.h"
#include <exception>
#include "CudaBsrMatrix.h"
#include <iostream>
#include "ldpdef.h"
#include <eigen\Sparse>
#include <eigen\Dense>

#define USE_CPU_PATH

#define CHECK(exp, msg)\
if (!(exp)){ throw std::exception(msg); }

#pragma region --cpu path
struct CudaCholeskeySolver_EigenContainter
{
	typedef float real;
	typedef Eigen::Matrix<real, -1, -1> Mat;
	typedef Eigen::SparseMatrix<real, Eigen::ColMajor> SpMat;
	typedef Eigen::Matrix<real, -1, 1> Vec;
	typedef Eigen::SimplicialLLT<SpMat, Eigen::Lower> Solver;

	SpMat A;
	Vec b;
	Vec x;
	Solver solver;
	std::vector<float> value_h;
	std::vector<float> xtmp;

	void analysis(const CudaBsrMatrix& dA)
	{
		value_h.resize(dA.nnz());

		A.resize(dA.rows(), dA.cols());
		A.resizeNonZeros(dA.nnz());
		x.resize(A.rows());
		b.resize(A.rows());
		xtmp.resize(A.rows());

		// copy structure from gpu
		cudaSafeCall(cudaMemcpy(A.outerIndexPtr(), dA.bsrRowPtr(), (dA.rows() + 1)*sizeof(int),
			cudaMemcpyDeviceToHost), "CudaCholeskeySolver::analysis() 1");
		cudaSafeCall(cudaMemcpy(A.innerIndexPtr(), dA.bsrColIdx(), dA.nnz()*sizeof(int),
			cudaMemcpyDeviceToHost), "CudaCholeskeySolver::analysis() 2");

		solver.analyzePattern(A.triangularView<Eigen::Lower>());
	}

	void factor(const CudaBsrMatrix& dA)
	{
		cudaSafeCall(cudaThreadSynchronize(), "CudaCholeskeySolver::factor() 0");
		cudaSafeCall(cudaMemcpy(value_h.data(), dA.value(), dA.nnz()*sizeof(float),
			cudaMemcpyDeviceToHost), "CudaCholeskeySolver::factor() 1");

		for (int r = 0; r < A.nonZeros(); r++)
			A.valuePtr()[r] = value_h[r];

		solver.factorize(A.triangularView<Eigen::Lower>());
	}

	void solve(float* d_x, const float* d_b)
	{
		cudaSafeCall(cudaMemcpy(xtmp.data(), d_b, A.rows()*sizeof(float),
			cudaMemcpyDeviceToHost), "CudaCholeskeySolver::solve() 1");
		for (int i = 0; i < A.rows(); i++)
			b[i] = xtmp[i];

		b = solver.permutationP()*b;
		solver.matrixL().solveInPlace(b);
		solver.matrixU().solveInPlace(b);
		x = solver.permutationPinv()*b;

		for (int i = 0; i < A.rows(); i++)
			xtmp[i] = x[i];
		cudaSafeCall(cudaMemcpy(d_x, xtmp.data(), A.rows()*sizeof(float),
			cudaMemcpyHostToDevice), "CudaCholeskeySolver::solve() 2");
	}

	// since A = L*L'
	// this functions solves for L*u = b
	void solveL(float* d_u, const float* d_b)
	{
		cudaSafeCall(cudaMemcpy(xtmp.data(), d_b, A.rows()*sizeof(float),
			cudaMemcpyDeviceToHost), "CudaCholeskeySolver::solveL() 1");
		for (int i = 0; i < A.rows(); i++)
			b[i] = xtmp[i];

		b = solver.permutationP()*b;
		solver.matrixL().solveInPlace(b);

		for (int i = 0; i < A.rows(); i++)
			xtmp[i] = b[i];
		cudaSafeCall(cudaMemcpy(d_u, xtmp.data(), A.rows()*sizeof(float),
			cudaMemcpyHostToDevice), "CudaCholeskeySolver::solveL() 2");
	}

	// this functions solves for L'*x = u
	void solveLt(float* d_x, const float* d_u)
	{
		cudaSafeCall(cudaMemcpy(xtmp.data(), d_u, A.rows()*sizeof(float),
			cudaMemcpyDeviceToHost), "CudaCholeskeySolver::solveLt() 1");
		for (int i = 0; i < A.rows(); i++)
			b[i] = xtmp[i];

		solver.matrixU().solveInPlace(b);
		x = solver.permutationPinv()*b;

		for (int i = 0; i < A.rows(); i++)
			xtmp[i] = x[i];
		cudaSafeCall(cudaMemcpy(d_x, xtmp.data(), A.rows()*sizeof(float),
			cudaMemcpyHostToDevice), "CudaCholeskeySolver::solveLt() 2");
	}

	static void dumpSparseMatrix(const SpMat& A, const char* filename)
	{
		FILE* pFile = fopen(filename, "w");
		if (!pFile)
			throw std::exception("dumpSparseMatrix: create file failed!");
		for (int r = 0; r < A.outerSize(); r++)
		{
			int rs = A.outerIndexPtr()[r];
			int re = A.outerIndexPtr()[r + 1];
			for (int c = rs; c < re; c++)
				fprintf(pFile, "%d %d %ef\n", r, A.innerIndexPtr()[c], A.valuePtr()[c]);
		}
		fclose(pFile);
	}
};
#pragma endregion


inline void cusolverCheck(cusolverStatus_t st, const char* msg)
{
	if (st != CUSOLVER_STATUS_SUCCESS)
	{
		std::cout << "[cusolver error( " << st << ")]: " << msg;
		throw std::exception();
	}
}

CudaCholeskeySolver::CudaCholeskeySolver()
{
	m_reorder = false;
	m_A = nullptr;
	m_A_csr = nullptr;
	m_handle = nullptr;
	m_info = nullptr;
	m_isAnalysied = false;
	m_isFactored = false;
	m_isInited = false;
	m_container = nullptr;
}

CudaCholeskeySolver::~CudaCholeskeySolver()
{
	clear();
}

void CudaCholeskeySolver::init()
{
#ifdef USE_CPU_PATH
	if (m_container == nullptr)
		m_container = new CudaCholeskeySolver_EigenContainter();
#else
	cusolverCheck(cusolverSpCreate(&m_handle), "cusolverSpCreate");
	cusolverCheck(cusolverSpCreateCsrcholInfo(&m_info), "cusolverSpCreateCsrcholInfo");
#endif
	
	m_isInited = true;
}

void CudaCholeskeySolver::clear()
{
	if (m_A_csr != m_A)
		delete m_A_csr;

#ifdef USE_CPU_PATH
	if (m_container)
		delete m_container;
#else
	m_workSpace.release();
	cusolverCheck(cusolverSpDestroyCsrcholInfo(m_info), "cusolverSpDestroyCsrcholInfo");
	cusolverCheck(cusolverSpDestroy(m_handle), "cusolverRfDestroy");
#endif
	m_reorder = false;
	m_A = nullptr;
	m_A_csr = nullptr;
	m_isAnalysied = false;
	m_isFactored = false;
	m_isInited = false;
}

// analysis the sparse pattern of A
void CudaCholeskeySolver::analysis(const CudaBsrMatrix* A, bool reorder)
{
	CHECK(m_isInited, "CudaCholeskeySolver::analysis(): call init() firstly");
	CHECK(A != nullptr, "CudaCholeskeySolver::analysis(): A nullptr");
	m_reorder = reorder;
	m_A = A;

	if (m_A->isCsr())
		m_A_csr = (CudaBsrMatrix*)m_A;
	else
	{
		if (m_A_csr == nullptr)
			m_A_csr = new CudaBsrMatrix(A->getCusparseHandle());
		m_A->toCsr_structure(*m_A_csr);
	}

#ifdef USE_CPU_PATH
	m_container->analysis(*m_A_csr);
#else
	// analysis csr pattern
	cusolverCheck(cusolverSpXcsrcholAnalysis(m_handle, m_A_csr->rows(), m_A_csr->nnz(),
		m_A_csr->getCuSparseMatDesc(), m_A_csr->bsrRowPtr(), m_A_csr->bsrColIdx(), m_info),
		"cusolverSpXcsrcholAnalysis");
	// allocate csr working buffer
	size_t bytes1 = 0, bytes2 = 0;
	cusolverCheck(cusolverSpScsrcholBufferInfo(m_handle, m_A_csr->rows(), m_A_csr->nnz(),
		m_A_csr->getCuSparseMatDesc(), m_A_csr->value(), m_A_csr->bsrRowPtr(),
		m_A_csr->bsrColIdx(), m_info, &bytes1, &bytes2), "cusolverSpScsrcholBufferInfo");
	if (bytes2 > m_workSpace.size())
		m_workSpace.create(bytes2 * 1.2);
#endif
	m_isAnalysied = true;
}

void CudaCholeskeySolver::factor()
{
	CHECK(m_isAnalysied, "CudaCholeskeySolver::factor(): call analysis() firstly");

	if (m_A != m_A_csr)
		m_A->toCsr_value(*m_A_csr);

#ifdef USE_CPU_PATH
	m_container->factor(*m_A_csr);
#else
	// csr factorize
	cusolverCheck(cusolverSpScsrcholFactor(m_handle, m_A_csr->rows(), m_A_csr->nnz(),
		m_A_csr->getCuSparseMatDesc(), m_A_csr->value(), m_A_csr->bsrRowPtr(), m_A_csr->bsrColIdx(),
		m_info, m_workSpace), "cusolverSpScsrcholFactor");
#endif

	m_isFactored = true;
}

void CudaCholeskeySolver::solve(float* x, const float* b)
{
	CHECK(m_isFactored, "CudaCholeskeySolver::solve(): call factor() firstly");

#ifdef USE_CPU_PATH
	m_container->solve(x, b);
#else
	// csr solve
	cusolverCheck(cusolverSpScsrcholSolve(m_handle, m_A_csr->rows(), b, x,
		m_info, m_workSpace.ptr()), "cusolverSpScsrcholSolve");	
#endif
}

// since A = L*L'
// this functions solves for L*u = b
void CudaCholeskeySolver::solveL(float* u, const float* b)
{
	CHECK(m_isFactored, "CudaCholeskeySolver::solveL(): call factor() firstly");

#ifdef USE_CPU_PATH
	m_container->solveL(u, b);
#else
	throw std::exception("CudaCholeskeySolver::solveL(): not implemented");
#endif
}

// this functions solves for L'*x = u
void CudaCholeskeySolver::solveLt(float* x, const float* u)
{
	CHECK(m_isFactored, "CudaCholeskeySolver::solveLt(): call factor() firstly");

#ifdef USE_CPU_PATH
	m_container->solveLt(x, u);
#else
	throw std::exception("CudaCholeskeySolver::solveLt(): not implemented");
#endif
}