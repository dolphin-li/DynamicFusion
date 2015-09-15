#pragma once

#define CONV_HELPER_ENABLE_SSE
#define CONV_HELPER_ENABLE_OMP 1

#include "util.h"

#ifdef CONV_HELPER_ENABLE_OMP
#include <omp.h>
#endif

#ifdef CONV_HELPER_ENABLE_SSE
#include <xmmintrin.h>      // __m128 data type and SSE functions
#endif

namespace conv_helper
{
	// 3D volume padding by zeros
	template<typename T, int N> void zero_padding3(T* dst, const T* src, ldp::Int3 srcRes)
	{
		ldp::Int3 dstRes = srcRes + N * 2;
		for (int z = 0; z < dstRes[2]; z++)
		{
			T* dst_z = dst + dstRes[0] * dstRes[1] * z;
			const T* src_z = src + srcRes[0] * srcRes[1] * (z - N);
			if (z < N || z >= srcRes[2] + N)
			{
				memset(dst_z, 0, dstRes[0] * dstRes[1] * sizeof(T));
				continue;
			}
			
			for (int y = 0; y < dstRes[1]; y++)
			{
				T* dst_y = dst_z + dstRes[0] * y;
				const T* src_y = src_z + srcRes[0] * (y - N);
				if (y < N || y >= srcRes[1] + N)
				{
					memset(dst_y, 0, dstRes[0] * sizeof(T));
					continue;
				}

				memset(dst_y, 0, N * sizeof(T));
				memset(dst_y + srcRes[0] + N, 0, N * sizeof(T));

				for (int x = N; x < dstRes[0] - N; x++)
					dst_y[x] = src_y[x - N];
			}// y
		}// z
	}

	// 1D max filter
	template<typename T, int N> void max_filter(T* dst, const T* src,
		int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		// the first few elements that does not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = std::numeric_limits<T>::lowest();
			for (int k = xb; k <= xe; k++)
			if (src[k + x] > v)
				v = src[k + x];
			*dst = v;
			dst += dstStride;
		}

		// middle elements that fullfills the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			T v = std::numeric_limits<T>::lowest();
			for (int k = -L; k <= R; k++)
			if (src[k + x] > v)
				v = src[k + x];
			*dst = v;
			dst += dstStride;
		}// end for x

		// the last few elements that does not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = std::numeric_limits<T>::lowest();
			for (int k = xb; k <= xe; k++)
			if (src[k + x] > v)
				v = src[k + x];
			*dst = v;
			dst += dstStride;
		}
	}

	template<typename T, int N> void min_filter(T* dst, const T* src,
		int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		// the first few elements that does not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = std::numeric_limits<T>::max();
			for (int k = xb; k <= xe; k++)
			if (src[k + x] < v)
				v = src[k + x];
			*dst = v;
			dst += dstStride;
		}

		// middle elements that fullfills the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			T v = std::numeric_limits<T>::max();
			for (int k = -L; k <= R; k++)
			if (src[k + x] < v)
				v = src[k + x];
			*dst = v;
			dst += dstStride;
		}// end for x

		// the last few elements that does not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = std::numeric_limits<T>::max();
			for (int k = xb; k <= xe; k++)
			if (src[k + x] < v)
				v = src[k + x];
			*dst = v;
			dst += dstStride;
		}
	}

#ifdef CONV_HELPER_ENABLE_SSE
	template<int N> void max_filter_sse(float* dst, const float* src,
		int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		__m128 s;

		// the first few elements that does not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::lowest());
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_max_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst += dstStride;
		}

		// middle elements that fullfills the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::lowest());
			for (int k = -L; k <= R; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_max_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst += dstStride;
		}// end for x

		// the last few elements that does not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::lowest());
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_max_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst += dstStride;
		}
	}

	template<int N> void min_filter_sse(float* dst, const float* src,
		int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		__m128 s;

		// the first few elements that does not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::max());
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_min_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst += dstStride;
		}

		// middle elements that fullfills the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::max());
			for (int k = -L; k <= R; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_min_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst += dstStride;
		}// end for x

		// the last few elements that does not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::max());
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_min_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst += dstStride;
		}
	}
#endif

	// 3D max filter
	// @dim: 
	//		0, filter x; 
	//		1, filter y; 
	//		2, filter z;
	//		-1[default], filter all directions
	template<typename T, int N, int numThreads = 4> void max_filter3(
		T* srcDst, ldp::Int3 res, int dim = -1)
	{
		T* dstPtr_000 = srcDst;
		const int y_stride = res[0];;
		const int z_stride = res[0] * res[1];

		if (dim < -1 || dim > 2)
			throw std::exception("illegal input parameter @dim");

		// allocate buffer for thread data
		std::vector<T> tmpBuffers[numThreads];
#ifdef CONV_HELPER_ENABLE_SSE
		std::vector<__m128> tmpBuffers_sse[numThreads];
#endif
		for (int k = 0; k < numThreads; k++)
		{
			tmpBuffers[k].resize(std::max(res[0], std::max(res[1], res[2])));
#ifdef CONV_HELPER_ENABLE_SSE
			if (ldp::is_float<T>::value)
				tmpBuffers_sse[k].resize(std::max(res[0], std::max(res[1], res[2])));
#endif
		}

		if (dim == 0 || dim == -1)
		{
			// max filtering along x direction
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
			for (int z = 0; z < res[2]; z++)
			{
				std::vector<T>& tmpBuffer = tmpBuffers[omp_get_thread_num()];
				T* dstPtr_00z = dstPtr_000 + z * z_stride;
				for (int y = 0; y < res[1]; y++)
				{
					T* dstPtr = dstPtr_00z + y * y_stride;
					for (int x = 0; x < res[0]; x++)
						tmpBuffer[x] = dstPtr[x];
					max_filter<T, N>(dstPtr, tmpBuffer.data(), res[0], 1);
				}// end for y
			}// end for z
		}// end if dim == 0

		if (dim == 1 || dim == -1)
		{
			// max filtering along y direction
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
			for (int z = 0; z < res[2]; z++)
			{
				std::vector<T>& tmpBuffer = tmpBuffers[omp_get_thread_num()];
				T* dstPtr_00z = dstPtr_000 + z * z_stride;
#ifdef CONV_HELPER_ENABLE_SSE
				std::vector<__m128>& tmpBuffer_sse =
					tmpBuffers_sse[omp_get_thread_num()];
#endif
				int x = 0;
#ifdef CONV_HELPER_ENABLE_SSE
				if (ldp::is_float<T>::value)
				{
					for (; x < res[0] - 3; x += 4)
					{
						T* dstPtr = dstPtr_00z + x;
						for (int y = 0, y1 = 0; y < res[1]; y++, y1 += y_stride)
							tmpBuffer_sse[y] = _mm_loadu_ps((float*)dstPtr + y1);
						max_filter_sse<N>((float*)dstPtr, (const float*)tmpBuffer_sse.data(), res[1], y_stride);
					}
				}
#endif
				for (; x < res[0]; x++)
				{
					T* dstPtr = dstPtr_00z + x;
					for (int y = 0, y1 = 0; y < res[1]; y++, y1 += y_stride)
						tmpBuffer[y] = dstPtr[y1];
					max_filter<T, N>(dstPtr, tmpBuffer.data(), res[1], y_stride);
				}// end for x
			}// end for z
		}// end if dim == 1

		if (dim == 2 || dim == -1)
		{
			// max filtering along z direction
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
			for (int y = 0; y < res[1]; y++)
			{
				std::vector<T>& tmpBuffer = tmpBuffers[omp_get_thread_num()];
				T* dstPtr_00y = dstPtr_000 + y * y_stride;
#ifdef CONV_HELPER_ENABLE_SSE
				std::vector<__m128>& tmpBuffer_sse =
					tmpBuffers_sse[omp_get_thread_num()];
#endif

				int x = 0;
#ifdef CONV_HELPER_ENABLE_SSE
				if (ldp::is_float<T>::value)
				{
					for (; x < res[0] - 3; x += 4)
					{
						T* dstPtr = dstPtr_00y + x;
						for (int z = 0, z1 = 0; z < res[2]; z++, z1 += z_stride)
							tmpBuffer_sse[z] = _mm_loadu_ps((float*)&dstPtr[z1]);
						max_filter_sse<N>((float*)dstPtr, (const float*)tmpBuffer_sse.data(), res[2], z_stride);
					}
				}
#endif
				for (; x < res[0]; x++)
				{
					T* dstPtr = dstPtr_00y + x;
					for (int z = 0, z1 = 0; z < res[2]; z++, z1 += z_stride)
						tmpBuffer[z] = dstPtr[z1];
					max_filter<T, N>(dstPtr, tmpBuffer.data(), res[2], z_stride);
				}// end for x
			}// end for y
		}// end if dim == 2
	}

	template<typename T, int N, int numThreads = 4> void min_filter3(
		T* srcDst, ldp::Int3 res, int dim = -1)
	{
		T* dstPtr_000 = srcDst;
		const int y_stride = res[0];;
		const int z_stride = res[0] * res[1];

		if (dim < -1 || dim > 2)
			throw std::exception("illegal input parameter @dim");

		// allocate buffer for thread data
		std::vector<T> tmpBuffers[numThreads];
#ifdef CONV_HELPER_ENABLE_SSE
		std::vector<__m128> tmpBuffers_sse[numThreads];
#endif
		for (int k = 0; k < numThreads; k++)
		{
			tmpBuffers[k].resize(std::max(res[0], std::max(res[1], res[2])));
#ifdef CONV_HELPER_ENABLE_SSE
			if (ldp::is_float<T>::value)
				tmpBuffers_sse[k].resize(std::max(res[0], std::max(res[1], res[2])));
#endif
		}

		if (dim == 0 || dim == -1)
		{
			// max filtering along x direction
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
			for (int z = 0; z < res[2]; z++)
			{
				std::vector<T>& tmpBuffer = tmpBuffers[omp_get_thread_num()];
				T* dstPtr_00z = dstPtr_000 + z * z_stride;
				for (int y = 0; y < res[1]; y++)
				{
					T* dstPtr = dstPtr_00z + y * y_stride;
					for (int x = 0; x < res[0]; x++)
						tmpBuffer[x] = dstPtr[x];
					min_filter<T, N>(dstPtr, tmpBuffer.data(), res[0], 1);
				}// end for y
			}// end for z
		}// end if dim == 0

		if (dim == 1 || dim == -1)
		{
			// max filtering along y direction
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
			for (int z = 0; z < res[2]; z++)
			{
				std::vector<T>& tmpBuffer = tmpBuffers[omp_get_thread_num()];
				T* dstPtr_00z = dstPtr_000 + z * z_stride;
#ifdef CONV_HELPER_ENABLE_SSE
				std::vector<__m128>& tmpBuffer_sse =
					tmpBuffers_sse[omp_get_thread_num()];
#endif
				int x = 0;
#ifdef CONV_HELPER_ENABLE_SSE
				if (ldp::is_float<T>::value)
				{
					for (; x < res[0] - 3; x += 4)
					{
						T* dstPtr = dstPtr_00z + x;
						for (int y = 0, y1 = 0; y < res[1]; y++, y1 += y_stride)
							tmpBuffer_sse[y] = _mm_loadu_ps((float*)dstPtr + y1);
						min_filter_sse<N>((float*)dstPtr, (const float*)tmpBuffer_sse.data(), res[1], y_stride);
					}
				}
#endif
				for (; x < res[0]; x++)
				{
					T* dstPtr = dstPtr_00z + x;
					for (int y = 0, y1 = 0; y < res[1]; y++, y1 += y_stride)
						tmpBuffer[y] = dstPtr[y1];
					min_filter<T, N>(dstPtr, tmpBuffer.data(), res[1], y_stride);
				}// end for x
			}// end for z
		}// end if dim == 1

		if (dim == 2 || dim == -1)
		{
			// max filtering along z direction
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
			for (int y = 0; y < res[1]; y++)
			{
				std::vector<T>& tmpBuffer = tmpBuffers[omp_get_thread_num()];
				T* dstPtr_00y = dstPtr_000 + y * y_stride;
#ifdef CONV_HELPER_ENABLE_SSE
				std::vector<__m128>& tmpBuffer_sse =
					tmpBuffers_sse[omp_get_thread_num()];
#endif

				int x = 0;
#ifdef CONV_HELPER_ENABLE_SSE
				if (ldp::is_float<T>::value)
				{
					for (; x < res[0] - 3; x += 4)
					{
						T* dstPtr = dstPtr_00y + x;
						for (int z = 0, z1 = 0; z < res[2]; z++, z1 += z_stride)
							tmpBuffer_sse[z] = _mm_loadu_ps((float*)&dstPtr[z1]);
						min_filter_sse<N>((float*)dstPtr, (const float*)tmpBuffer_sse.data(), res[2], z_stride);
					}
				}
#endif
				for (; x < res[0]; x++)
				{
					T* dstPtr = dstPtr_00y + x;
					for (int z = 0, z1 = 0; z < res[2]; z++, z1 += z_stride)
						tmpBuffer[z] = dstPtr[z1];
					min_filter<T, N>(dstPtr, tmpBuffer.data(), res[2], z_stride);
				}// end for x
			}// end for y
		}// end if dim == 2
	}

	// 1D conv, the same with matlab conv(..., 'same')
	//	assume:
	//		the stride of src is 1 
	//		the size of src @num
	//		the size of dst @num
	//		kernel size is @N
	template<typename T, int N> void conv(T* dst, const T* src, const T* kernel,
		int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		// the first few elements that does not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = 0;
			for (int k = xb; k <= xe; k++)
				v += src[k + x] * kernel[R - k];
			*dst = v;
			dst += dstStride;
		}

		// middle elements that fullfills the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			T v = 0;
			for (int k = -L; k <= R; k++)
				v += src[k + x] * kernel[R - k];
			*dst = v;
			dst += dstStride;
		}// end for x

		// the last few elements that does not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = 0;
			for (int k = xb; k <= xe; k++)
				v += src[k + x] * kernel[R - k];
			*dst = v;
			dst += dstStride;
		}
	}

#ifdef CONV_HELPER_ENABLE_SSE
	template<int N> void conv_sse(float* dst, const float* src, const float* kernel,
		int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		__m128 s, knl;

		// the first few elements that do not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_setzero_ps();
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				knl = _mm_set_ps1(kernel[R - k]);
				v = _mm_add_ps(v, _mm_mul_ps(s, knl));
			}
			_mm_storeu_ps(dst, v);
			dst += dstStride;
		}

		// middle elements that fullfill the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			__m128 v = _mm_setzero_ps();
			for (int k = -L; k <= R; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				knl = _mm_set_ps1(kernel[R - k]);
				v = _mm_add_ps(v, _mm_mul_ps(s, knl));
			}
			_mm_storeu_ps(dst, v);
			dst += dstStride;
		}// end for x

		// the last few elements that do not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_setzero_ps();
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				knl = _mm_set_ps1(kernel[R - k]);
				v = _mm_add_ps(v, _mm_mul_ps(s, knl));
			}
			_mm_storeu_ps(dst, v);
			dst += dstStride;
		}
	}
#endif
	// 3D seperate-kernel convolution with 'same' output
	// this method is the SAME as calling matlab convn(...,'same') along x-y-z 3 dims.
	// @data:
	//	X-Y-Z ordered 3D data
	// @kernel[N]
	// @res: resolution of the input 3D data
	// @dim: 
	//		0, conv x; 
	//		1, conv y; 
	//		2, conv z;
	//		-1[default], conv all directions
	template<typename T, int N, int numThreads = 4> void conv3(T* srcDst, 
		const T* kernel, ldp::Int3 res, int dim = -1)
	{
		T* dstPtr_000 = srcDst;
		const int y_stride = res[0];;
		const int z_stride = res[0]*res[1];

		if (dim < -1 || dim > 2)
			throw std::exception("illegal input parameter @dim");

		// allocate buffer for thread data
		std::vector<T> tmpBuffers[numThreads];
#ifdef CONV_HELPER_ENABLE_SSE
		std::vector<__m128> tmpBuffers_sse[numThreads];
#endif
		for (int k = 0; k < numThreads; k++)
		{
			tmpBuffers[k].resize(std::max(res[0], std::max(res[1], res[2])));
#ifdef CONV_HELPER_ENABLE_SSE
			if (ldp::is_float<T>::value)
				tmpBuffers_sse[k].resize(std::max(res[0], std::max(res[1], res[2])));
#endif
		}

		if (dim == 0 || dim == -1)
		{
			// conv along x direction
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
			for (int z = 0; z < res[2]; z++)
			{
				std::vector<T>& tmpBuffer = tmpBuffers[omp_get_thread_num()];
				T* dstPtr_00z = dstPtr_000 + z * z_stride;

				for (int y = 0; y < res[1]; y++)
				{
					T* dstPtr = dstPtr_00z + y * y_stride;
					for (int x = 0; x < res[0]; x++)
						tmpBuffer[x] = dstPtr[x];
					conv<T, N>(dstPtr, tmpBuffer.data(), kernel, res[0], 1);
				}// end for y
			}// end for z
		}// end if dim == 0

		if (dim == 1 || dim == -1)
		{
			// conv along y direction
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
			for (int z = 0; z < res[2]; z++)
			{
				std::vector<T>& tmpBuffer = tmpBuffers[omp_get_thread_num()];
#ifdef CONV_HELPER_ENABLE_SSE
				std::vector<__m128>& tmpBuffer_sse =
					tmpBuffers_sse[omp_get_thread_num()];
#endif
				T* dstPtr_00z = dstPtr_000 + z * z_stride;

				int x = 0;
#ifdef CONV_HELPER_ENABLE_SSE
				if (ldp::is_float<T>::value)
				{
					for (; x < res[0] - 3; x += 4)
					{
						T* dstPtr = dstPtr_00z + x;
						for (int y = 0, y1 = 0; y < res[1]; y++, y1 += y_stride)
							tmpBuffer_sse[y] = _mm_loadu_ps((float*)dstPtr + y1);
						conv_sse<N>(dstPtr, (const float*)tmpBuffer_sse.data(), kernel, res[1], y_stride);
					}
				}
#endif
				for (; x < res[0]; x++)
				{
					T* dstPtr = dstPtr_00z + x;
					for (int y = 0, y1 = 0; y < res[1]; y++, y1 += y_stride)
						tmpBuffer[y] = dstPtr[y1];
					conv<T, N>(dstPtr, tmpBuffer.data(), kernel, res[1], y_stride);
				}// end for x
			}// end for z
		}// end if dim == 1


		if (dim == 2 || dim == -1)
		{
			// conv along z direction
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
			for (int y = 0; y < res[1]; y++)
			{
				std::vector<T>& tmpBuffer = tmpBuffers[omp_get_thread_num()];
#ifdef CONV_HELPER_ENABLE_SSE
				std::vector<__m128>& tmpBuffer_sse =
					tmpBuffers_sse[omp_get_thread_num()];
#endif
				T* dstPtr_00y = dstPtr_000 + y * y_stride;

				int x = 0;
#ifdef CONV_HELPER_ENABLE_SSE
				if (ldp::is_float<T>::value)
				{
					for (; x < res[0] - 3; x += 4)
					{
						T* dstPtr = dstPtr_00y + x;
						for (int z = 0, z1 = 0; z < res[2]; z++, z1 += z_stride)
							tmpBuffer_sse[z] = _mm_loadu_ps((float*)&dstPtr[z1]);
						conv_sse<N>((float*)dstPtr, (const float*)tmpBuffer_sse.data(), kernel, res[2], z_stride);
					}
				}
#endif
				for (; x < res[0]; x++)
				{
					T* dstPtr = dstPtr_00y + x;
					for (int z = 0, z1 = 0; z < res[2]; z++, z1 += z_stride)
						tmpBuffer[z] = dstPtr[z1];
					conv<T, N>(dstPtr, tmpBuffer.data(), kernel, res[2], z_stride);
				}// end for x
			}// end for y
		}// end if dim == 2
	}


	// 3D box filter
	// it is done by integral images
	// temporary memroy will be allocated inside
	template<typename T, int numThreads = 4> void boxFilter(T* dst, const T*src, int boxSize, ldp::Int3 res)
	{
		if (dst == src)
			throw std::exception("boxFilter(): src and dst cannot be the same memory!");
		std::vector<T> intImg((1+res[0]) * (1+res[1]) * (1+res[2]), 0);
		const static int L = boxSize / 2 - (boxSize % 2 == 0);
		const static int R = boxSize / 2;
		const int stride_z_intg = (res[0] + 1)*(res[1] + 1);
		const int stride_z_srcDst = res[0] * res[1];

		// integral along z
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
		for (int y = 0; y < res[1]; y++)
		{
			const T* src_y = src + y*res[0];
			T* intg_y = intImg.data() + (y + 1)*(res[0] + 1);
			for (int z = 0; z < res[2]; z++)
			{
				const T* src_z = src_y + z*stride_z_srcDst;
				T* intg_z = intg_y + (z + 1)*stride_z_intg;
				const T* intg_z_prev = intg_z - stride_z_intg;
				for (int x = 0; x < res[0]; x++)
					intg_z[x + 1] = intg_z_prev[x + 1] + src_z[x];
			}// z
		}// y

		// diff along z
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
		for (int y = 0; y < res[1]; y++)
		{
			T* dst_y = dst + y*res[0];
			const T* intg_y = intImg.data() + (y + 1)*(res[0] + 1);
			for (int z = 0; z < res[2]; z++)
			{
				T* dst_z = dst_y + z*stride_z_srcDst;
				const T* intg_z_after = intg_y + std::min(z + R + 1, res[2])*stride_z_intg;
				const T* intg_z_prev = intg_y + std::max(z - L, 0)*stride_z_intg;
				for (int x = 0; x < res[0]; x++)
					dst_z[x] = intg_z_after[x + 1] - intg_z_prev[x + 1];
			}// z
		}// y

		// integral along y
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
		for (int z = 0; z < res[2]; z++)
		{
			const T* dst_z = dst + z*stride_z_srcDst;
			T* intg_z = intImg.data() + (z + 1)*stride_z_intg;
			for (int y = 0; y < res[1]; y++)
			{
				const T* dst_y = dst_z + y*res[0];
				T* intg_y = intg_z + (y+1)*(res[0]+1);
				const T* intg_y_prev = intg_y - (res[0]+1);
				for (int x = 0; x < res[0]; x++)
					intg_y[x+1] = intg_y_prev[x+1] + dst_y[x];
			}// y
		}// z

		// diff along y
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
		for (int z = 0; z < res[2]; z++)
		{
			T* dst_z = dst + z*stride_z_srcDst;
			const T* intg_z = intImg.data() + (z + 1)*stride_z_intg;
			for (int y = 0; y < res[1]; y++)
			{
				T* dst_y = dst_z + y*res[0];
				const T* intg_y_after = intg_z + std::min(y + R + 1, res[1])*(res[0] + 1);
				const T* intg_y_prev = intg_z + std::max(y - L, 0)*(res[0] + 1);
				for (int x = 0; x < res[0]; x++)
					dst_y[x] = intg_y_after[x + 1] - intg_y_prev[x + 1];
			}// y
		}// z

		// integral along x
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
		for (int z = 0; z < res[2]; z++)
		{
			const T* dst_z = dst + z*stride_z_srcDst;
			T* intg_z = intImg.data() + (z + 1)*stride_z_intg;
			for (int y = 0; y < res[1]; y++)
			{
				const T* dst_y = dst_z + y*res[0];
				T* intg_y = intg_z + (y + 1)*(res[0] + 1);
				for (int x = 0; x < res[0]; x++)
					intg_y[x+1] = intg_y[x] + dst_y[x];
			}// y
		}// z

		// diff along x
		const int pos[] = { std::min(L, res[0]), std::max(0, res[0] - R - 1) };
#pragma omp parallel for num_threads(numThreads) if(CONV_HELPER_ENABLE_OMP)
		for (int z = 0; z < res[2]; z++)
		{
			T* dst_z = dst + z*stride_z_srcDst;
			const T* intg_z = intImg.data() + (z + 1)*stride_z_intg;
			for (int y = 0; y < res[1]; y++)
			{
				T* dst_y = dst_z + y*res[0];
				const T* intg_y = intg_z + (y + 1)*(res[0] + 1);
				for (int x = 0; x < pos[0]; x++)
				{
					int prev = std::max(x - L, 0);
					int after = std::min(x + R + 1, res[0]);
					dst_y[x] = intg_y[after] - intg_y[prev];
				}
				for (int x = pos[0]; x < pos[1]; x++)
					dst_y[x] = intg_y[x + R + 1] - intg_y[x - L];
				for (int x = pos[1]; x < res[0]; x++)
				{
					int prev = std::max(x - L, 0);
					int after = std::min(x + R + 1, res[0]);
					dst_y[x] = intg_y[after] - intg_y[prev];
				}
			}// y
		}// z
	}
}