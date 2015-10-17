#pragma once

namespace gpu_cholesky
{
#if defined(__CUDACC__)
	// row majored matrix
	__device__ __forceinline__ int lex_index_2D(int r, int c, int row_length)
	{
		return c + r*row_length;
	}

	__device__ __forceinline__ float sqr(float a)
	{
		return a*a;
	}

	template<class T, int N> struct SmallDeviceMat
	{
		__device__ SmallDeviceMat(T* d):A(d){}
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

	// batched cholesky decomposition of PSD matrix:
	// ptr: input/output, only the lower triangular part is read and wrote.
	template<typename T, int N>
	__global__ void __single_thread_cholesky_batched(T *ptr, int stride, int batchSize)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if(tid >= batchSize)
			return;
		SmallDeviceMat<T,N> A(ptr + tid*stride);
		for (unsigned int r = 0; r < N; ++r)
		{
			T sum = A(r, r);
			for (unsigned int u = 0; u < r; ++u)
				sum -= sqr(A(r, u));
			T Lii = sqrt(sum);
			A(r, r) = Lii;
			for (unsigned int c = r + 1; c < N; ++c)
			{
				sum = A(c, r) ;
				for (unsigned int u = 0; u < r; ++u)
					sum -= A(c, u) * A(r, u);
				A(c, r) = sum / Lii;
			}
		}
	}


	// batched triangular matrix inverse of Lower part:
	// ptr: input/output, only the lower triangular part is read and wrote.
	template<typename T, int N>
	__global__ void __single_thread_tril_inv_batched(T *ptr, int stride, int batchSize)
	{
		int tid = threadIdx.x + blockIdx.x*blockDim.x;
		if(tid >= batchSize)
			return;
		SmallDeviceMat<T, N> A(ptr + tid*stride);

		for (int i = 0; i < N; ++i) 
		{
			A(i,i) = 1.f / A(i,i);
			for (int j = i + 1; j < N; j++) 
			{
				float sum = 0.f;
				for (int k = i; k < j; k++)
					sum -= A(j,k) * A(k,i);
				A(j,i) = sum / A(j,j);
			}
		}
	}


	// batched triangular matrix multiplication: L'*L
	// ptr: input/output, only the lower triangular part is read and wrote.
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
			for(int x=0; x<=y; ++x)
			{
				T sum = 0.f;
				for(int k=y; k<N; k++)
					sum += A(k, y) * A(k, x);
				C(y, x) = C(x, y) = sum;
			}
		}
	}
#endif
}