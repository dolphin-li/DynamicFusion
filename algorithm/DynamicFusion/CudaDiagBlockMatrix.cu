#include "CudaDiagBlockMatrix.h"

__global__ void CudaDiagBlockMatrix_set(int n, float* ptr, float val)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
		ptr[i] = val;
}

__global__ void CudaDiagBlockMatrix_scale_add(int n, float* ptr, float alpha, float beta)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
		ptr[i] = alpha * ptr[i] + beta;
}

__global__ void CudaDiagBlockMatrix_scale_add_diag(int nRow, int bSz, float* ptr, float alpha, float beta)
{
	int iDiag = threadIdx.x + blockIdx.x * blockDim.x;
	if (iDiag < nRow)
	{
		int iBlock = iDiag / bSz;
		int shift = iDiag - iBlock * bSz;
		int pos = (iBlock*bSz + shift)*bSz + shift;
		ptr[pos] = alpha * ptr[pos] + beta;
	}
}


// vec_out = alpha * Linv * vec_in + beta * vec_out
__global__ void CudaDiagBlockMatrix_Lv_kernel(
	cudaTextureObject_t tex,
	float* vec_out, const float* vec_in, 
	int nRows, int nodePerRow, float alpha, float beta)
{
	int iRow = threadIdx.x + blockIdx.x*blockDim.x;
	if (iRow >= nRows)
		return;
	int iNode = iRow / nodePerRow;
	int rshift = iRow - iNode * nodePerRow;
	int iPos = iRow * nodePerRow + rshift;

	float sum = 0.f;
	for (int k = -rshift; k <= 0; k++)
	{
		float val = 0.f;
		tex1Dfetch(&val, tex, iPos + k);
		sum += val * vec_in[iRow + k];
	}

	vec_out[iRow] = alpha * sum + beta * vec_out[iRow];
}

// vec_out = alpha * Ltinv * vec_in + beta * vec_out
__global__ void CudaDiagBlockMatrix_Ltv_kernel(
	cudaTextureObject_t tex,
	float* vec_out, const float* vec_in, 
	int nRows, int nodePerRow, float alpha, float beta)
{
	int iRow = threadIdx.x + blockIdx.x*blockDim.x;
	if (iRow >= nRows)
		return;
	int iNode = iRow / nodePerRow;
	int rshift = iRow - iNode * nodePerRow;
	int iPos = iRow * nodePerRow + rshift;

	float sum = 0.f;
	for (int k = 0; k < nodePerRow - rshift; k++)
	{
		float val = 0.f;
		tex1Dfetch(&val, tex, iPos + k*nodePerRow);
		sum += val * vec_in[iRow + k];
	}

	vec_out[iRow] = alpha * sum + beta * vec_out[iRow];
}

__global__ void CudaDiagBlockMatrix_transpose_L_to_U(float* Hd, int nnz, int rows, int varPerNode)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nnz)
		return;
	int varPerNode2 = varPerNode*varPerNode;
	int iNode = tid / varPerNode2;
	int pos = iNode * varPerNode2;
	int shift = tid - pos;
	int rowShift = shift / varPerNode;
	int colShift = shift - rowShift * varPerNode;

	if (rowShift < colShift)
		Hd[tid] = Hd[pos + colShift * varPerNode + rowShift];
}

CudaDiagBlockMatrix& CudaDiagBlockMatrix::operator = (float constVal)
{
	if (nnz() == 0)
		return *this;
	if (constVal == 0.f)
	{
		cudaSafeCall(cudaMemset(m_values.ptr(), 0, nnz()*m_values.elem_size),
			"CudaDiagBlockMatrix::operator = 0");
	}
	else
	{
		CudaDiagBlockMatrix_set << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(nnz(), m_values.ptr(), constVal);
		cudaSafeCall(cudaGetLastError(), "CudaDiagBlockMatrix::operator = constVal");
	}
	return *this;
}

CudaDiagBlockMatrix& CudaDiagBlockMatrix::axpy(float alpha, float beta)
{
	if (nnz() == 0)
		return *this;
	CudaDiagBlockMatrix_scale_add << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(nnz(), m_values.ptr(), alpha, beta);
	cudaSafeCall(cudaGetLastError(), "CudaDiagBlockMatrix::axpy");
	return *this;
}

CudaDiagBlockMatrix& CudaDiagBlockMatrix::axpy_diag(float alpha, float beta)
{
	if (rows() == 0)
		return *this;
	CudaDiagBlockMatrix_scale_add_diag << <divUp(rows(), CTA_SIZE), CTA_SIZE >> >(
		rows(), blockSize(), m_values.ptr(), alpha, beta);
	cudaSafeCall(cudaGetLastError(), "CudaDiagBlockMatrix::axpy_diag");
	return *this;
}

CudaDiagBlockMatrix& CudaDiagBlockMatrix::transpose_L_to_U()
{
	if (nnz() == 0)
		return *this;
	CudaDiagBlockMatrix_transpose_L_to_U << <divUp(nnz(), CTA_SIZE), CTA_SIZE >> >(
		value(), nnz(), rows(), blockSize());
	cudaSafeCall(cudaGetLastError(), "CudaDiagBlockMatrix::transpose_L_to_U");
	return *this;
}

// vec_out = alpha * Lower(this) * vec_in + beta;
void CudaDiagBlockMatrix::Lv(const float* vec_in, float* vec_out, float alpha, float beta)
{
	if (rows() == 0)
		return;
	CudaDiagBlockMatrix_Lv_kernel << <divUp(rows(), CTA_SIZE), CTA_SIZE >> >(
		m_tex, vec_out, vec_in, rows(), blockSize(), alpha, beta);
	cudaSafeCall(cudaGetLastError(), "CudaDiagBlockMatrix::Lv");
}

// vec_out = alpha * Lower(this)^t * vec_in + beta;
void CudaDiagBlockMatrix::Ltv(const float* vec_in, float* vec_out, float alpha, float beta)
{
	if (rows() == 0)
		return;
	CudaDiagBlockMatrix_Ltv_kernel << <divUp(rows(), CTA_SIZE), CTA_SIZE >> >(
		m_tex, vec_out, vec_in, rows(), blockSize(), alpha, beta);
	cudaSafeCall(cudaGetLastError(), "CudaDiagBlockMatrix::Ltv");
}