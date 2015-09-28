#include "ModerGpuWrapper.h"
#include "moderngpu\include\util\mgpucontext.h"
#include "moderngpu\include\moderngpu.cuh"
#include <algorithm>
#include <device_functions.h>
#include <helper_math.h>
using namespace mgpu;


namespace modergpu_wrapper
{
	ContextPtr g_context;

	static void create_context()
	{
		if (g_context.get() == nullptr)
			g_context = CreateCudaDevice(0);
	}

	void mergesort(int* val, int count)
	{
		create_context();
		MergesortKeys(val, count, *g_context);
	}

	void mergesort(float* val, int count)
	{
		create_context();
		MergesortKeys(val, count, *g_context);
	}

	void mergesort_by_key(int* keys_global, int* values_global, int count)
	{
		create_context();
		MergesortPairs(keys_global, values_global, count, *g_context);
	}
	void mergesort_by_key(int* keys_global, float* values_global, int count)
	{
		create_context();
		MergesortPairs(keys_global, values_global, count, *g_context);
	}
	void mergesort_by_key(float* keys_global, int* values_global, int count)
	{
		create_context();
		MergesortPairs(keys_global, values_global, count, *g_context);
	}
	void mergesort_by_key(float* keys_global, float* values_global, int count)
	{
		create_context();
		MergesortPairs(keys_global, values_global, count, *g_context);
	}
	void mergesort_by_key(int* keys_global, float4* values_global, int count)
	{
		create_context();
		MergesortPairs(keys_global, values_global, count, *g_context);
	}

	void inclusive_scan_by_key(int* key_d, float4* value_d, float4* dst_d, int n)
	{

	}
}