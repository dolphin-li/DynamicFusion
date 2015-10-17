#include "GpuCholeSky.h"
#include <cuda.h>
#include "cuda_utils.h"
#include <exception>
namespace gpu_cholesky
{
	enum
	{
		TILE_SIZE = 32
	};

	__device__ int lex_index_2D(int r, int c, int row_length)
	{
		return c + r*row_length;
	}
	__device__ int global_pos(int t_pos, int block_offset)
	{
		return t_pos + TILE_SIZE*block_offset;
	}

	// row majored matrix
	__device__ __forceinline__ float sqr(float a)
	{
		return a*a;
	}

	template<class T, int N> struct SmallDeviceMat
	{
		__device__ SmallDeviceMat(T* d) :A(d){}
		T* A;
		__device__ __forceinline__ T& operator()(int r, int c)
		{
			return A[r*N + c];
		}
		__device__ __forceinline__ const T& operator()(int r, int c)const
		{
			return A[r*N + c];
		}
	};

	template<typename T, int N>
	__global__ void __single_thread_cholesky_batched(T *ptr, int stride, int batchSize)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid >= batchSize)
			return;
		SmallDeviceMat<T, N> A(ptr + tid*stride);
		for (unsigned int r = 0; r < N; ++r)
		{
			T sum = A(r, r);
			for (unsigned int u = 0; u < r; ++u)
				sum -= sqr(A(r, u));
			T Lii = sqrt(sum);
			A(r, r) = Lii;
			for (unsigned int c = r + 1; c < N; ++c)
			{
				sum = A(c, r);
				for (unsigned int u = 0; u < r; ++u)
					sum -= A(c, u) * A(r, u);
				A(c, r) = sum / Lii;
			}
		}
	}


	template<typename T, int N>
	__global__ void __single_thread_tril_inv_batched(T *ptr, int stride, int batchSize)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid >= batchSize)
			return;
		SmallDeviceMat<T, N> A(ptr + tid*stride);

		for (int i = 0; i < N; ++i)
		{
			A(i, i) = 1.f / A(i, i);
			for (int j = i + 1; j < N; j++)
			{
				float sum = 0.f;
				for (int k = i; k < j; k++)
					sum -= A(j, k) * A(k, i);
				A(j, i) = sum / A(j, j);
			}
		}
	}


	template<typename T, int N>
	__global__ void __single_thread_LtL_batched(T* outputLLt, int strideLLt,
		const T *inputL, int strideL, int batchSize)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if (tid >= batchSize)
			return;
		SmallDeviceMat<T, N> A((T*)inputL + tid*strideL);
		SmallDeviceMat<T, N> C(outputLLt + tid*strideLLt);

		for (int y = 0; y < N; ++y)
		{
			for (int x = 0; x <= y; ++x)
			{
				T sum = 0.f;
				for (int k = y; k<N; k++)
					sum += A(k, y) * A(k, x);
				C(y, x) = C(x, y) = sum;
			}
		}
	}

	template<typename T>
	__global__ void __factorize_diagonal_block(T *A, int block_offset,
		int global_row_length)
	{
		int col = threadIdx.x;

		int row = threadIdx.y;

		int global_row = global_pos(row, block_offset);
		int global_col = global_pos(col, block_offset);
		int idx = lex_index_2D(global_row, global_col, global_row_length);

		__shared__ T L[TILE_SIZE][TILE_SIZE + 1];

		L[row][col] = A[idx];
		__syncthreads();

		T fac;

		for (int k = 0; k < TILE_SIZE; k++)
		{
			__syncthreads();
			fac = rsqrtf(L[k][k]);
			__syncthreads();

			if ((row == k) && (col >= k)) L[col][row] = (L[col][row])*fac;

			__syncthreads();


			if ((row >= col) && (col > k)) L[row][col] -= L[col][k] * L[row][k];
		}

		__syncthreads();

		if (row >= col) A[idx] = L[row][col];
	}
	template<typename T>
	__global__ void __strip_update(T *A, int block_offset, int global_row_length)
	{
		int boffy = block_offset;
		int boffx = blockIdx.x + boffy + 1;

		int col = threadIdx.x;
		int row = threadIdx.y;

		__shared__ T topleft[TILE_SIZE][TILE_SIZE + 1];
		__shared__ T workingmat[TILE_SIZE][TILE_SIZE + 1];

		int global_row = global_pos(row, block_offset);
		int global_col = global_pos(col, block_offset);

		int idx = lex_index_2D(global_row, global_col, global_row_length);

		topleft[row][col] = A[idx];

		global_row = global_pos(row, boffx);
		int idx_w = lex_index_2D(global_row, global_col, global_row_length);
		workingmat[col][row] = A[idx_w];

		__syncthreads();

		if (row == 0)
		for (int k = 0; k < TILE_SIZE; k++)
		{
			T sum = 0.;
			for (int m = 0; m < k; m++) sum += topleft[k][m] * workingmat[m][col];

			workingmat[k][col] = (workingmat[k][col] - sum) / topleft[k][k];
		}

		__syncthreads();

		A[idx_w] = workingmat[col][row];
	}
	template<typename T>
	__global__ void __diag_update(T *A, int block_offset, int global_row_length)
	{
		int boffx = blockIdx.x + block_offset + 1;

		int col = threadIdx.x;
		int row = threadIdx.y;

		int global_row = global_pos(row, boffx);
		int global_col = global_pos(col, block_offset);

		int idx = lex_index_2D(global_row, global_col, global_row_length);

		__shared__ T left[TILE_SIZE][TILE_SIZE + 1];

		left[row][col] = A[idx];

		__syncthreads();

		T sum = 0.f;

		if (row >= col)
		{
			for (int kk = 0; kk < TILE_SIZE; kk++) sum += left[row][kk] * left[col][kk];

			global_col = global_pos(col, boffx);
			idx = lex_index_2D(global_row, global_col, global_row_length);

			A[idx] -= sum;
		}
	}
	template<typename T>
	__global__ void __lo_update(T *A, int block_offset, int n_blocks, int global_row_length)
	{
		int col = threadIdx.x;
		int row = threadIdx.y;

		int boffy = blockIdx.y + block_offset + 1;
		int boffx = boffy + 1;

		__shared__ T left[TILE_SIZE][TILE_SIZE];

		__shared__ T upt[TILE_SIZE][TILE_SIZE + 1];


		int global_row = global_pos(row, boffy);
		int global_col = global_pos(col, block_offset);

		int idx = lex_index_2D(global_row, global_col, global_row_length);

		upt[row][col] = A[idx];

		for (; boffx < n_blocks; boffx++)
		{
			global_row = global_pos(row, boffx);
			idx = lex_index_2D(global_row, global_col, global_row_length);

			left[row][col] = A[idx];

			__syncthreads();

			T matrixprod = 0.f;

			for (int kk = 0; kk < TILE_SIZE; kk++)
				matrixprod += left[row][kk] * upt[col][kk];

			__syncthreads();

			global_col = global_pos(col, boffy);
			idx = lex_index_2D(global_row, global_col, global_row_length);

			A[idx] -= matrixprod;
		}
	}

	template<typename T, int N>
	void _single_thread_cholesky_batched(T *ptr, int stride, int batchSize)
	{
		dim3 block(TILE_SIZE);
		dim3 grid(divUp(batchSize, block.x));
		__single_thread_cholesky_batched<T, N> << <grid, block >> >(
			ptr, stride, batchSize);
		cudaSafeCall(cudaGetLastError(), "__single_thread_cholesky_batched");
	}

	template<typename T, int N>
	void _single_thread_tril_inv_batched(T *ptr, int stride, int batchSize)
	{
		dim3 block(TILE_SIZE);
		dim3 grid(divUp(batchSize, block.x));
		__single_thread_tril_inv_batched<T, N> << <grid, block >> >(
			ptr, stride, batchSize);
		cudaSafeCall(cudaGetLastError(), "__single_thread_tril_inv_batched");
	}

	template<typename T, int N>
	void _single_thread_LtL_batched(T* outputLLt, int strideLLt,
		const T *inputL, int strideL, int batchSize)
	{
		dim3 block(TILE_SIZE);
		dim3 grid(divUp(batchSize, block.x));
		__single_thread_LtL_batched<T, N> << <grid, block >> >(
			outputLLt, strideLLt, inputL, strideL, batchSize);
		cudaSafeCall(cudaGetLastError(), "__single_thread_LtL_batched");
	}

	template<typename T>
	cudaError_t _factorize_diagonal_block(T * a_d, int block_offset, int n_rows_padded)
	{
		dim3 threads(TILE_SIZE, TILE_SIZE);
		__factorize_diagonal_block << <1, threads >> >(a_d, block_offset, n_rows_padded);
		cudaThreadSynchronize();

		return cudaGetLastError();
	}

	template<typename T>
	void _strip_update(T *a_d, int block_offset, int n_remaining_blocks, int n_rows_padded)
	{
		cudaError_t error;

		dim3 stripgrid(n_remaining_blocks - 1);
		dim3 threads(TILE_SIZE, TILE_SIZE);
		__strip_update << <stripgrid, threads >> >(a_d, block_offset, n_rows_padded);
		cudaThreadSynchronize();
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			printf("     Error code %d: %s.\n", error, cudaGetErrorString(error));
			exit(-1);
		}
	}

	template<typename T>
	void _diag_update(T *a_d, int block_offset, int n_rows_padded, int n_remaining_blocks)
	{
		cudaError_t error;
		dim3 stripgrid(n_remaining_blocks - 1);
		dim3 threads(TILE_SIZE, TILE_SIZE);
		__diag_update << <stripgrid, threads >> >(a_d, block_offset, n_rows_padded);
		cudaThreadSynchronize();
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			printf("     Error code %d: %s.\n", error, cudaGetErrorString(error));
			exit(-1);
		}
	}

	template<typename T>
	void _lo_update(T *a_d, int block_offset, int n_blocks, int n_rows_padded, int n_remaining_blocks)
	{
		cudaError_t error;
		dim3 logrid;
		logrid.x = 1;
		logrid.y = n_remaining_blocks - 2;
		dim3 threads(TILE_SIZE, TILE_SIZE);
		__lo_update << < logrid, threads >> >(a_d, block_offset, n_blocks, n_rows_padded);
		cudaThreadSynchronize();
		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			printf("     Error code %d: %s.\n", error, cudaGetErrorString(error));
			exit(-1);
		}
	}

	template<typename T>
	void _cholesky(T * a_d, int n_rows)
	{
		cudaError_t error;

		int n_blocks = (n_rows + int(TILE_SIZE) - 1) / int(TILE_SIZE);
		int n_rows_padded = n_blocks*TILE_SIZE;

		dim3 threads(TILE_SIZE, TILE_SIZE);

		dim3 logrid;

		for (int i = n_blocks; i > 2; --i)
		{
			logrid.x = 1;
			logrid.y = i - 2;

			dim3 stripgrid(i - 1);

			__factorize_diagonal_block << <1, threads >> >(a_d, n_blocks - i, n_rows_padded);
			cudaThreadSynchronize();

			__strip_update << <stripgrid, threads >> >(a_d, n_blocks - i, n_rows_padded);
			cudaThreadSynchronize();

			__diag_update << <stripgrid, threads >> >(a_d, n_blocks - i, n_rows_padded);
			cudaThreadSynchronize();
			__lo_update << < logrid, threads >> >(a_d, n_blocks - i, n_blocks, n_rows_padded);
			cudaThreadSynchronize();
		}

		if (n_blocks > 1)
		{
			__factorize_diagonal_block << <1, threads >> >(a_d, n_blocks - 2,
				n_rows_padded);
			cudaThreadSynchronize();

			__strip_update << <1, threads >> >(a_d, n_blocks - 2, n_rows_padded);
			cudaThreadSynchronize();

			__diag_update << <1, threads >> >(a_d, n_blocks - 2, n_rows_padded);
			cudaThreadSynchronize();
		}

		__factorize_diagonal_block << <1, threads >> >(a_d, n_blocks - 1, n_rows_padded);

		cudaThreadSynchronize();

		error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			printf("     Error code %d: %s.\n", error, cudaGetErrorString(error));
			exit(-1);
		}
	}

	cudaError_t factorize_diagonal_block(float *A, int block_offset, int global_row_length)
	{
		return _factorize_diagonal_block(A, block_offset, global_row_length);
	}

	void strip_update(float *A, int block_offset, int n_remaining_blocks, int n_rows_padded)
	{
		_strip_update(A, block_offset, n_remaining_blocks, n_rows_padded);
	}

	void diag_update(float *A, int block_offset, int global_row_length, int n_remaining_blocks)
	{
		_diag_update(A, block_offset, global_row_length, n_remaining_blocks);
	}

	void lo_update(float *A, int block_offset, int n_blocks, int global_row_length, int n_remaining_blocks)
	{
		_lo_update(A, block_offset, n_blocks, global_row_length, n_remaining_blocks);
	}
	void cholesky(float * a_d, int n_rows) { _cholesky(a_d, n_rows); }

	void single_thread_cholesky_batched(float *ptr, int nMatRowCol, int stride, int batchSize)
	{
		switch (nMatRowCol)
		{
		case 0:
			return;
		case 1:
			_single_thread_cholesky_batched<float, 1>(ptr, stride, batchSize);
			break;
		case 2:
			_single_thread_cholesky_batched<float, 2>(ptr, stride, batchSize);
			break;
		case 3:
			_single_thread_cholesky_batched<float, 3>(ptr, stride, batchSize);
			break;
		case 4:
			_single_thread_cholesky_batched<float, 4>(ptr, stride, batchSize);
			break;
		case 5:
			_single_thread_cholesky_batched<float, 5>(ptr, stride, batchSize);
			break;
		case 6:
			_single_thread_cholesky_batched<float, 6>(ptr, stride, batchSize);
			break;
		case 7:
			_single_thread_cholesky_batched<float, 7>(ptr, stride, batchSize);
			break;
		case 8:
			_single_thread_cholesky_batched<float, 8>(ptr, stride, batchSize);
			break;
		case 9:
			_single_thread_cholesky_batched<float, 9>(ptr, stride, batchSize);
			break;
		default:
			throw std::exception("error: non-supported size in _single_thread_cholesky_batched");
		}
	}

	void single_thread_tril_inv_batched(float *ptr, int nMatRowCol, int stride, int batchSize)
	{
		switch (nMatRowCol)
		{
		case 0:
			return;
		case 1:
			_single_thread_tril_inv_batched<float, 1>(ptr, stride, batchSize);
			break;
		case 2:
			_single_thread_tril_inv_batched<float, 2>(ptr, stride, batchSize);
			break;
		case 3:
			_single_thread_tril_inv_batched<float, 3>(ptr, stride, batchSize);
			break;
		case 4:
			_single_thread_tril_inv_batched<float, 4>(ptr, stride, batchSize);
			break;
		case 5:
			_single_thread_tril_inv_batched<float, 5>(ptr, stride, batchSize);
			break;
		case 6:
			_single_thread_tril_inv_batched<float, 6>(ptr, stride, batchSize);
			break;
		case 7:
			_single_thread_tril_inv_batched<float, 7>(ptr, stride, batchSize);
			break;
		case 8:
			_single_thread_tril_inv_batched<float, 8>(ptr, stride, batchSize);
			break;
		case 9:
			_single_thread_tril_inv_batched<float, 9>(ptr, stride, batchSize);
			break;
		default:
			throw std::exception("error: non-supported size in single_thread_tril_inv_batched");
		}
	}

	void single_thread_LtL_batched(float* outputLLt, int strideLLt,
		const float *inputL, int strideL, int nMatRowCol, int batchSize)
	{
		switch (nMatRowCol)
		{
		case 0:
			return;
		case 1:
			_single_thread_LtL_batched<float, 1>(outputLLt, strideLLt, inputL, strideL, batchSize);
			break;
		case 2:
			_single_thread_LtL_batched<float, 2>(outputLLt, strideLLt, inputL, strideL, batchSize);
			break;
		case 3:
			_single_thread_LtL_batched<float, 3>(outputLLt, strideLLt, inputL, strideL, batchSize);
			break;
		case 4:
			_single_thread_LtL_batched<float, 4>(outputLLt, strideLLt, inputL, strideL, batchSize);
			break;
		case 5:
			_single_thread_LtL_batched<float, 5>(outputLLt, strideLLt, inputL, strideL, batchSize);
			break;
		case 6:
			_single_thread_LtL_batched<float, 6>(outputLLt, strideLLt, inputL, strideL, batchSize);
			break;
		case 7:
			_single_thread_LtL_batched<float, 7>(outputLLt, strideLLt, inputL, strideL, batchSize);
			break;
		case 8:
			_single_thread_LtL_batched<float, 8>(outputLLt, strideLLt, inputL, strideL, batchSize);
			break;
		case 9:
			_single_thread_LtL_batched<float, 9>(outputLLt, strideLLt, inputL, strideL, batchSize);
			break;
		default:
			throw std::exception("error: non-supported size in single_thread_LtL_batched");
		}
	}

}