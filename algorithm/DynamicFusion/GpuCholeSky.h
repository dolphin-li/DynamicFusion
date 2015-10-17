#pragma once

namespace gpu_cholesky
{
#if defined(__CUDACC__)
	// row majored matrix
	__device__ __forceinline__ int lex_index_2D(int r, int c, int row_length)
	{
		return c + r*row_length;
	}

	template<typename T, int N>
	__global__ void __single_thread_cholesky(T *A)
	{
		for (unsigned int r = 0; r < N; ++r)
		{
			T sum = 0.;
			unsigned int idx;
			unsigned int idx_c;
			for (unsigned int u = 0; u < r; ++u)
			{
				idx = lex_index_2D(r, u, N);
				sum += A[idx] * A[idx];
			}
			idx = lex_index_2D(r, r, N);
			A[idx] = sqrt(A[idx] - sum);

			for (unsigned int c = r + 1; c < N; ++c)
			{
				sum = 0.;

				for (unsigned int u = 0; u < r; ++u)
				{
					idx_c = lex_index_2D(c, u, N);
					idx = lex_index_2D(r, u, N);
					sum += A[idx_c] * A[idx];
				}

				idx_c = lex_index_2D(c, r, N);
				idx = lex_index_2D(r, c, N);
				A[idx_c] = A[idx] - sum;

				idx = lex_index_2D(r, r, N);
				A[idx_c] /= A[idx];
			}
		}
	}
#endif
}