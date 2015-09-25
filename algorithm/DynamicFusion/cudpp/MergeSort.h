#pragma once
#include <device_functions.h>
#include <driver_functions.h>
#include <vector_functions.h>
#include "cuda_utils.h"
#include "device_array.h"
namespace dfusion
{
	void mergeSort(const int* key_in, const int* val_in, int* key_out, int* val_out, int n, bool less = 1);

	void mergeSort(const int* key_in, const float* val_in, int* key_out, float* val_out, int n, bool less = 1);

	void mergeSort(const float* key_in, const int* val_in, float* key_out, int* val_out, int n, bool less = 1);
}