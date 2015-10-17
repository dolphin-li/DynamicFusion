#pragma once

namespace gpu_cholesky
{
	void cholesky(float * A, int n_rows);

	// batched cholesky decomposition of PSD matrix:
	// ptr: input/output, only the lower triangular part is read and wrote.
	void single_thread_LtL_batched(float* outputLLt, int strideLLt,
		const float *inputL, int strideL, int nMatRowCol, int batchSize);

	// batched triangular matrix inverse of Lower part:
	// ptr: input/output, only the lower triangular part is read and wrote.
	void single_thread_tril_inv_batched(float *ptr, int nMatRowCol, int stride, int batchSize);

	// batched triangular matrix multiplication: L'*L
	// ptr: input/output, only the lower triangular part is read and wrote.
	void single_thread_cholesky_batched(float *ptr, int nMatRowCol, int stride, int batchSize);
}