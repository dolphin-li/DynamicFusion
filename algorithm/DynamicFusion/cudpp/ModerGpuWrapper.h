#pragma once

namespace modergpu_wrapper
{
	void mergesort(int* val, int count);
	void mergesort(float* val, int count);
	void mergesort_by_key(int* keys_global, int* values_global, int count);
	void mergesort_by_key(int* keys_global, float* values_global, int count);
	void mergesort_by_key(float* keys_global, int* values_global, int count);
	void mergesort_by_key(float* keys_global, float* values_global, int count);
	void mergesort_by_key(int* keys_global, float4* values_global, int count);
}