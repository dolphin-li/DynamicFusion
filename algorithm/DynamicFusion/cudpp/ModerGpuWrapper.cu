#include "ModerGpuWrapper.h"
#include "moderngpu\include\util\mgpucontext.h"
#include "moderngpu\include\moderngpu.cuh"
#include <algorithm>

using namespace mgpu;


namespace modergpu_wrapper
{
	ContextPtr g_context = CreateCudaDevice(0);

	void mergesort_by_key(int* keys_global, int* values_global, int count)
	{
		MergesortPairs(keys_global, values_global, count, *g_context);
	}
	void mergesort_by_key(int* keys_global, float* values_global, int count)
	{
		MergesortPairs(keys_global, values_global, count, *g_context);
	}
	void mergesort_by_key(float* keys_global, int* values_global, int count)
	{
		MergesortPairs(keys_global, values_global, count, *g_context);
	}
	void mergesort_by_key(float* keys_global, float* values_global, int count)
	{
		MergesortPairs(keys_global, values_global, count, *g_context);
	}
}