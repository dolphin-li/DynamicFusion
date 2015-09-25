#pragma once

#include "cudpp.h"
#include "cudpp_hash.h"

namespace cudpp_wrapper
{
	void exlusive_scan(unsigned int* d_in, unsigned int* d_out, int n);

	void merge_sort_by_key(int* keys_d, float* values_d, int n);
	void merge_sort_by_key(float* keys_d, int* values_d, int n);
}