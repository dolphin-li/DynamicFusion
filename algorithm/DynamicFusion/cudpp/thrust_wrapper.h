#pragma once

#include <device_functions.h>
#include <driver_functions.h>
#include <vector_functions.h>
#include <cuda_fp16.h>

namespace thrust_wrapper
{
	void sort_by_key(int* key_d, float4* value_d, int n);
	void sort_by_key(float* key_d, int* value_d, int n);
	void sort_by_key(int* key_d, int* value_d, int n);
	void sort_by_key(int* key_d, float* value_d, int n);

	void stable_sort_by_key(int* key_d, float4* value_d, int n);
	void stable_sort_by_key(float* key_d, int* value_d, int n);

	void exclusive_scan(const int* in, int* out, int n);
	void exclusive_scan(const unsigned int* in, unsigned int* out, int n);

	void inclusive_scan_by_key(int* key_d, float4* value_d, float4* dst_d, int n);

	size_t unique_by_key(int* key_d, float4* value_d, int n);

	// the input value are counting iterators from @input_count_begin
	size_t unique_counting_by_key_copy(const int* key_d, int input_count_begin, 
		int* out_key_d, int* out_value_d, int n);

	// out[map[i]] = in[i];
	void scatter(const int* in_d, const int* map_d, int* out_d, int n);

	// out[i] = in[map[i])
	void gather(const int* in_d, const int* map_d, int* out_d, int n); 
	void gather(const float4* in_d, const int* map_d, float4* out_d, int n);

	// xyzwxyzw...xyzw -> xxx...yyy...zzz
	void seperate_channels(const float4* xyzw_d, float* x_d, float* y_d, float* z_d, int n);

	// ptr_d[i] = begin+i
	void make_counting_array(int* ptr_d, int n, int begin=0);

	void copy(int* dst, const int* src, int size);
	void copy(float* dst, const float* src, int size);

	// ptr_d[i] = value
	void assign(int* ptr_d, int value, int n);
	void assign(float* ptr_d, float value, int n);
	void assign(float4* ptr_d, float4 value, int n);
}