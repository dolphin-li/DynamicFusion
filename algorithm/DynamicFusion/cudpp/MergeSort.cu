
#include <assert.h>
#include "MergeSort.h"

namespace dfusion
{
#define SHARED_SIZE_LIMIT 1024U
#define     SAMPLE_STRIDE 128
	typedef unsigned int uint;
#define MAX_SAMPLE_COUNT 32768
#define W (sizeof(uint) * 8)



	static inline __host__ __device__ uint iDivUp(uint a, uint b)
	{
		return ((a % b) == 0) ? (a / b) : (a / b + 1);
	}

	static inline __host__ __device__ uint getSampleCount(uint dividend)
	{
		return iDivUp(dividend, SAMPLE_STRIDE);
	}

	static inline __device__ uint nextPowerOfTwo(uint x)
	{
		/*
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return ++x;
		*/
		return 1U << (W - __clz(x - 1));
	}

#pragma region MergeSort
	template<class KeyType, uint sortDir> static inline __device__ uint binarySearchInclusive(
		KeyType val, KeyType *data, uint L, uint stride)
	{
		if (L == 0)
		{
			return 0;
		}

		uint pos = 0;

		for (; stride > 0; stride >>= 1)
		{
			uint newPos = min(pos + stride, L);

			if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)))
			{
				pos = newPos;
			}
		}

		return pos;
	}

	template<class KeyType, uint sortDir> static inline __device__ uint binarySearchExclusive(
		KeyType val, KeyType *data, uint L, uint stride)
	{
		if (L == 0)
		{
			return 0;
		}

		uint pos = 0;

		for (; stride > 0; stride >>= 1)
		{
			uint newPos = min(pos + stride, L);

			if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)))
			{
				pos = newPos;
			}
		}

		return pos;
	}

	template<class KeyType, class ValueType, uint sortDir> __global__ void mergeSortSharedKernel(
		KeyType *d_DstKey,
		ValueType *d_DstVal,
		KeyType *d_SrcKey,
		ValueType *d_SrcVal,
		uint arrayLength
		)
	{
		__shared__ KeyType s_key[SHARED_SIZE_LIMIT];
		__shared__ ValueType s_val[SHARED_SIZE_LIMIT];

		d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
		s_key[threadIdx.x + 0] = d_SrcKey[0];
		s_val[threadIdx.x + 0] = d_SrcVal[0];
		s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
		s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

		for (uint stride = 1; stride < arrayLength; stride <<= 1)
		{
			uint     lPos = threadIdx.x & (stride - 1);
			KeyType *baseKey = s_key + 2 * (threadIdx.x - lPos);
			ValueType *baseVal = s_val + 2 * (threadIdx.x - lPos);

			__syncthreads();
			KeyType keyA = baseKey[lPos + 0];
			ValueType valA = baseVal[lPos + 0];
			KeyType keyB = baseKey[lPos + stride];
			ValueType valB = baseVal[lPos + stride];
			uint posA = binarySearchExclusive<KeyType, sortDir>(keyA, baseKey + stride, stride, stride) + lPos;
			uint posB = binarySearchInclusive<KeyType, sortDir>(keyB, baseKey + 0, stride, stride) + lPos;

			__syncthreads();
			baseKey[posA] = keyA;
			baseVal[posA] = valA;
			baseKey[posB] = keyB;
			baseVal[posB] = valB;
		}

		__syncthreads();
		d_DstKey[0] = s_key[threadIdx.x + 0];
		d_DstVal[0] = s_val[threadIdx.x + 0];
		d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
		d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
	}

	template<class KeyType, uint sortDir> __global__ void generateSampleRanksKernel(
		uint *d_RanksA,
		uint *d_RanksB,
		KeyType *d_SrcKey,
		uint stride,
		uint N,
		uint threadCount
		)
	{
		uint pos = blockIdx.x * blockDim.x + threadIdx.x;

		if (pos >= threadCount)
		{
			return;
		}

		const uint i = pos & ((stride / SAMPLE_STRIDE) - 1);
		const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
		d_SrcKey += segmentBase;
		d_RanksA += segmentBase / SAMPLE_STRIDE;
		d_RanksB += segmentBase / SAMPLE_STRIDE;

		const uint segmentElementsA = stride;
		const uint segmentElementsB = min(stride, N - segmentBase - stride);
		const uint segmentSamplesA = getSampleCount(segmentElementsA);
		const uint segmentSamplesB = getSampleCount(segmentElementsB);

		if (i < segmentSamplesA)
		{
			d_RanksA[i] = i * SAMPLE_STRIDE;
			d_RanksB[i] = binarySearchExclusive<KeyType, sortDir>(
				d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride,
				segmentElementsB, nextPowerOfTwo(segmentElementsB)
				);
		}

		if (i < segmentSamplesB)
		{
			d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
			d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive<KeyType, sortDir>(
				d_SrcKey[stride + i * SAMPLE_STRIDE], d_SrcKey + 0,
				segmentElementsA, nextPowerOfTwo(segmentElementsA)
				);
		}
	}


	__global__ void mergeRanksAndIndicesKernel(
		uint *d_Limits,
		uint *d_Ranks,
		uint stride,
		uint N,
		uint threadCount
		)
	{
		uint pos = blockIdx.x * blockDim.x + threadIdx.x;

		if (pos >= threadCount)
		{
			return;
		}

		const uint i = pos & ((stride / SAMPLE_STRIDE) - 1);
		const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
		d_Ranks += (pos - i) * 2;
		d_Limits += (pos - i) * 2;

		const uint segmentElementsA = stride;
		const uint segmentElementsB = min(stride, N - segmentBase - stride);
		const uint  segmentSamplesA = getSampleCount(segmentElementsA);
		const uint  segmentSamplesB = getSampleCount(segmentElementsB);

		if (i < segmentSamplesA)
		{
			uint dstPos = binarySearchExclusive<uint, 1U>(d_Ranks[i], d_Ranks + segmentSamplesA,
				segmentSamplesB, nextPowerOfTwo(segmentSamplesB)) + i;
			d_Limits[dstPos] = d_Ranks[i];
		}

		if (i < segmentSamplesB)
		{
			uint dstPos = binarySearchInclusive<uint, 1U>(d_Ranks[segmentSamplesA + i], d_Ranks,
				segmentSamplesA, nextPowerOfTwo(segmentSamplesA)) + i;
			d_Limits[dstPos] = d_Ranks[segmentSamplesA + i];
		}
	}


	template<class KeyType, class ValueType, uint sortDir> inline __device__ void merge(
		KeyType *dstKey,
		ValueType *dstVal,
		KeyType *srcAKey,
		ValueType *srcAVal,
		KeyType *srcBKey,
		ValueType *srcBVal,
		uint lenA,
		uint nPowTwoLenA,
		uint lenB,
		uint nPowTwoLenB
		)
	{
		KeyType keyA, keyB;
		ValueType valA, valB;
		uint dstPosA, dstPosB;

		if (threadIdx.x < lenA)
		{
			keyA = srcAKey[threadIdx.x];
			valA = srcAVal[threadIdx.x];
			dstPosA = binarySearchExclusive<KeyType, sortDir>(keyA, srcBKey, lenB, nPowTwoLenB) + threadIdx.x;
		}

		if (threadIdx.x < lenB)
		{
			keyB = srcBKey[threadIdx.x];
			valB = srcBVal[threadIdx.x];
			dstPosB = binarySearchInclusive<KeyType, sortDir>(keyB, srcAKey, lenA, nPowTwoLenA) + threadIdx.x;
		}

		__syncthreads();

		if (threadIdx.x < lenA)
		{
			dstKey[dstPosA] = keyA;
			dstVal[dstPosA] = valA;
		}

		if (threadIdx.x < lenB)
		{
			dstKey[dstPosB] = keyB;
			dstVal[dstPosB] = valB;
		}
	}

	template<class KeyType, class ValueType, uint sortDir> __global__ void mergeElementaryIntervalsKernel(
		KeyType *d_DstKey,
		ValueType *d_DstVal,
		KeyType *d_SrcKey,
		ValueType *d_SrcVal,
		uint *d_LimitsA,
		uint *d_LimitsB,
		uint stride,
		uint N
		)
	{
		__shared__ KeyType s_key[2 * SAMPLE_STRIDE];
		__shared__ ValueType s_val[2 * SAMPLE_STRIDE];

		const uint intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
		const uint segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
		d_SrcKey += segmentBase;
		d_SrcVal += segmentBase;
		d_DstKey += segmentBase;
		d_DstVal += segmentBase;

		//Set up threadblock-wide parameters
		__shared__ uint startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

		if (threadIdx.x == 0)
		{
			uint segmentElementsA = stride;
			uint segmentElementsB = umin(stride, N - segmentBase - stride);
			uint segmentSamplesA = getSampleCount(segmentElementsA);
			uint segmentSamplesB = getSampleCount(segmentElementsB);
			uint segmentSamples = segmentSamplesA + segmentSamplesB;

			startSrcA = d_LimitsA[blockIdx.x];
			startSrcB = d_LimitsB[blockIdx.x];
			uint endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
			uint endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
			lenSrcA = endSrcA - startSrcA;
			lenSrcB = endSrcB - startSrcB;
			startDstA = startSrcA + startSrcB;
			startDstB = startDstA + lenSrcA;
		}

		//Load main input data
		__syncthreads();

		if (threadIdx.x < lenSrcA)
		{
			s_key[threadIdx.x + 0] = d_SrcKey[0 + startSrcA + threadIdx.x];
			s_val[threadIdx.x + 0] = d_SrcVal[0 + startSrcA + threadIdx.x];
		}

		if (threadIdx.x < lenSrcB)
		{
			s_key[threadIdx.x + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + threadIdx.x];
			s_val[threadIdx.x + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + threadIdx.x];
		}

		//Merge data in shared memory
		__syncthreads();
		merge<KeyType, ValueType, sortDir>(
			s_key,
			s_val,
			s_key + 0,
			s_val + 0,
			s_key + SAMPLE_STRIDE,
			s_val + SAMPLE_STRIDE,
			lenSrcA, SAMPLE_STRIDE,
			lenSrcB, SAMPLE_STRIDE
			);

		//Store merged data
		__syncthreads();

		if (threadIdx.x < lenSrcA)
		{
			d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x];
			d_DstVal[startDstA + threadIdx.x] = s_val[threadIdx.x];
		}

		if (threadIdx.x < lenSrcB)
		{
			d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x];
			d_DstVal[startDstB + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
		}
	}

	template<class KeyType, class ValueType> class MergeSort
	{
		uint *d_RanksA, *d_RanksB, *d_LimitsA, *d_LimitsB;
		DeviceArray<char> merge_sort_buf;
		////////////////////////////////////////////////////////////////////////////////
		// Bottom-level merge sort (binary search-based)
		////////////////////////////////////////////////////////////////////////////////
		void mergeSortShared(
			KeyType *d_DstKey,
			ValueType *d_DstVal,
			KeyType *d_SrcKey,
			ValueType *d_SrcVal,
			uint batchSize,
			uint arrayLength,
			uint sortDir
			)
		{
			if (arrayLength < 2)
			{
				return;
			}

			assert(SHARED_SIZE_LIMIT % arrayLength == 0);
			assert(((batchSize * arrayLength) % SHARED_SIZE_LIMIT) == 0);
			uint  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
			uint threadCount = SHARED_SIZE_LIMIT / 2;

			if (sortDir)
			{
				mergeSortSharedKernel<KeyType, ValueType, 1U> << <blockCount, threadCount >> >(
					d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
				cudaSafeCall(cudaGetLastError(), "mergeSortShared<1><<<>>> failed\n");
			}
			else
			{
				mergeSortSharedKernel<KeyType, ValueType, 0U> << <blockCount, threadCount >> >(
					d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
				cudaSafeCall(cudaGetLastError(), "mergeSortShared<0><<<>>> failed\n");
			}
		}


		////////////////////////////////////////////////////////////////////////////////
		// Merge step 1: generate sample ranks
		////////////////////////////////////////////////////////////////////////////////

		void generateSampleRanks(
			uint *d_RanksA,
			uint *d_RanksB,
			KeyType *d_SrcKey,
			uint stride,
			uint N,
			uint sortDir
			)
		{
			uint lastSegmentElements = N % (2 * stride);
			uint threadCount = (lastSegmentElements > stride) ?
				(N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) :
				(N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

			if (sortDir)
			{
				generateSampleRanksKernel<KeyType, 1U> << <iDivUp(threadCount, 256), 256 >> >(
					d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
				cudaSafeCall(cudaGetLastError(), "generateSampleRanksKernel<1U><<<>>> failed\n");
			}
			else
			{
				generateSampleRanksKernel<KeyType, 0U> << <iDivUp(threadCount, 256), 256 >> >(
					d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
				cudaSafeCall(cudaGetLastError(), "generateSampleRanksKernel<0U><<<>>> failed\n");
			}
		}

		////////////////////////////////////////////////////////////////////////////////
		// Merge step 2: generate sample ranks and indices
		////////////////////////////////////////////////////////////////////////////////

		void mergeRanksAndIndices(
			uint *d_LimitsA,
			uint *d_LimitsB,
			uint *d_RanksA,
			uint *d_RanksB,
			uint stride,
			uint N
			)
		{
			uint lastSegmentElements = N % (2 * stride);
			uint threadCount = (lastSegmentElements > stride) ?
				(N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) :
				(N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

			mergeRanksAndIndicesKernel << <iDivUp(threadCount, 256), 256 >> >(
				d_LimitsA,
				d_RanksA,
				stride,
				N,
				threadCount
				);
			cudaSafeCall(cudaGetLastError(), "mergeRanksAndIndicesKernel(A)<<<>>> failed\n");

			mergeRanksAndIndicesKernel << <iDivUp(threadCount, 256), 256 >> >(
				d_LimitsB,
				d_RanksB,
				stride,
				N,
				threadCount
				);
			cudaSafeCall(cudaGetLastError(), "mergeRanksAndIndicesKernel(B)<<<>>> failed\n");
		}

		////////////////////////////////////////////////////////////////////////////////
		// Merge step 3: merge elementary intervals
		////////////////////////////////////////////////////////////////////////////////

		void mergeElementaryIntervals(
			KeyType *d_DstKey,
			ValueType *d_DstVal,
			KeyType *d_SrcKey,
			ValueType *d_SrcVal,
			uint *d_LimitsA,
			uint *d_LimitsB,
			uint stride,
			uint N,
			uint sortDir
			)
		{
			uint lastSegmentElements = N % (2 * stride);
			uint mergePairs = (lastSegmentElements > stride) ? getSampleCount(N) :
				(N - lastSegmentElements) / SAMPLE_STRIDE;

			if (sortDir)
			{
				mergeElementaryIntervalsKernel<KeyType, ValueType, 1U> << <mergePairs, SAMPLE_STRIDE >> >(
					d_DstKey,
					d_DstVal,
					d_SrcKey,
					d_SrcVal,
					d_LimitsA,
					d_LimitsB,
					stride,
					N
					);
				cudaSafeCall(cudaGetLastError(), "mergeElementaryIntervalsKernel<1> failed\n");
			}
			else
			{
				mergeElementaryIntervalsKernel<KeyType, ValueType, 0U> << <mergePairs, SAMPLE_STRIDE >> >(
					d_DstKey,
					d_DstVal,
					d_SrcKey,
					d_SrcVal,
					d_LimitsA,
					d_LimitsB,
					stride,
					N
					);
				cudaSafeCall(cudaGetLastError(), "mergeElementaryIntervalsKernel<0> failed\n");
			}
		}


		void initMergeSort(void)
		{
			cudaSafeCall(cudaMalloc((void **)&d_RanksA, MAX_SAMPLE_COUNT * sizeof(uint)));
			cudaSafeCall(cudaMalloc((void **)&d_RanksB, MAX_SAMPLE_COUNT * sizeof(uint)));
			cudaSafeCall(cudaMalloc((void **)&d_LimitsA, MAX_SAMPLE_COUNT * sizeof(uint)));
			cudaSafeCall(cudaMalloc((void **)&d_LimitsB, MAX_SAMPLE_COUNT * sizeof(uint)));
		}

		void closeMergeSort(void)
		{
			cudaSafeCall(cudaFree(d_RanksA));
			cudaSafeCall(cudaFree(d_RanksB));
			cudaSafeCall(cudaFree(d_LimitsB));
			cudaSafeCall(cudaFree(d_LimitsA));
		}

	public:
		MergeSort()
		{
			initMergeSort();
		}
		~MergeSort()
		{
			closeMergeSort();
		}
		void mergeSort(
			KeyType *d_DstKey,
			ValueType *d_DstVal,
			KeyType *d_SrcKey,
			ValueType *d_SrcVal,
			uint N,
			uint sortDir
			)
		{
			if (N*(sizeof(KeyType)+sizeof(ValueType)) > merge_sort_buf.size())
				merge_sort_buf.create(N*(sizeof(ValueType)+sizeof(KeyType))*1.5);

			KeyType* d_BufKey = (KeyType*)merge_sort_buf.ptr();
			ValueType* d_BufVal = (ValueType*)(merge_sort_buf.ptr()+N*sizeof(KeyType));

			uint stageCount = 0;

			for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1, stageCount++);

			KeyType *ikey, *okey;
			ValueType *ival, *oval;

			if (stageCount & 1)
			{
				ikey = d_BufKey;
				ival = d_BufVal;
				okey = d_DstKey;
				oval = d_DstVal;
			}
			else
			{
				ikey = d_DstKey;
				ival = d_DstVal;
				okey = d_BufKey;
				oval = d_BufVal;
			}

			assert(N <= (SAMPLE_STRIDE * MAX_SAMPLE_COUNT));
			assert(N % SHARED_SIZE_LIMIT == 0);
			mergeSortShared(ikey, ival, d_SrcKey, d_SrcVal, N / SHARED_SIZE_LIMIT, SHARED_SIZE_LIMIT, sortDir);

			for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1)
			{
				uint lastSegmentElements = N % (2 * stride);

				//Find sample ranks and prepare for limiters merge
				generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, N, sortDir);

				//Merge ranks and indices
				mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride, N);

				//Merge elementary intervals
				mergeElementaryIntervals(okey, oval, ikey, ival, d_LimitsA, d_LimitsB, stride, N, sortDir);

				if (lastSegmentElements <= stride)
				{
					//Last merge segment consists of a single array which just needs to be passed through
					cudaSafeCall(cudaMemcpy(okey + (N - lastSegmentElements), ikey + (N - lastSegmentElements), lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice));
					cudaSafeCall(cudaMemcpy(oval + (N - lastSegmentElements), ival + (N - lastSegmentElements), lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice));
				}

				KeyType *t;
				t = ikey;
				ikey = okey;
				okey = t;
				ValueType* t1;
				t1 = ival;
				ival = oval;
				oval = t1;
			}
		}
	};

	//MergeSort<int, int> g_merge_sort_i_i;
	//MergeSort<int, float> g_merge_sort_i_f;
	//MergeSort<float, int> g_merge_sort_f_i;
	//DeviceArray<char> g_merge_sort_buf;

	void mergeSort(const int* key_in, const int* val_in, int* key_out, int* val_out, int n, bool less)
	{
		//g_merge_sort_i_i.mergeSort((int*)key_out, (int*)val_out, (int*)key_in, (int*)val_in, n, less);
	}
	void mergeSort(const int* key_in, const float* val_in, int* key_out, float* val_out, int n, bool less)
	{
		//g_merge_sort_i_f.mergeSort((int*)key_out, (float*)val_out, (int*)key_in, (float*)val_in, n, less);
	}
	void mergeSort(const float* key_in, const int* val_in, float* key_out, int* val_out, int n, bool less)
	{
		//g_merge_sort_f_i.mergeSort((float*)key_out, (int*)val_out, (float*)key_in, (int*)val_in, n, less);
	}
#pragma endregion

}