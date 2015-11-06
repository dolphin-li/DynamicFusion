#pragma once

#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverRf.h>
#include <device_array.h>

class CudaBsrMatrix;
struct CudaCholeskeySolver_EigenContainter;
class CudaCholeskeySolver
{
public:
	CudaCholeskeySolver();
	~CudaCholeskeySolver();

	void init();
	void clear();

	// analysis the sparse pattern of A
	// if reorder, then a CPU-path reorder algorithm is performed
	//				which needs cpu-gpu data copies.
	//				but the fill-ins will be greatly reduced when factor
	void analysis(const CudaBsrMatrix* A, bool reorder = false);

	// numerical factorize
	void factor();

	// solve for A*x = b
	void solve(float* x, const float* b);

	// since A = L*L'
	// this functions solves for L*u = b
	void solveL(float* u, const float* b);

	// this functions solves for L'*x = u
	void solveLt(float* x, const float* u);
private:
	const CudaBsrMatrix* m_A;
	CudaBsrMatrix* m_A_csr;
	bool m_reorder;
	bool m_isAnalysied;
	bool m_isFactored;
	bool m_isInited;

	cusolverSpHandle_t m_handle;
	csrcholInfo_t m_info;
	DeviceArray<char> m_workSpace;

	CudaCholeskeySolver_EigenContainter* m_container;
};

