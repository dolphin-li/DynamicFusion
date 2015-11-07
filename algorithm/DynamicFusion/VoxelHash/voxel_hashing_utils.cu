/***********************************************************/
/**	\file
	\brief		util functions that must be compiled in nvcc
	\details	
	\author		Yizhong Zhang
	\date		12/7/2013
*/
/***********************************************************/
#include <thrust/scan.h>

#include "voxel_hashing_internal.h"


/**
	alloc memory for gpu storage
*/
extern "C"
void initMemoryc(thrust::device_vector<char>& vec, int size){
	vec.resize(size);
}

extern "C"
void initMemoryf(thrust::device_vector<float>& vec, int size){
	vec.resize(size);
}

extern "C"
void initMemoryi(thrust::device_vector<int>& vec, int size){
	vec.resize(size);
}

extern "C"
void initMemoryull(thrust::device_vector<unsigned long long>& vec, int size){
	vec.resize(size);
}


/**
	perform scan on a vector
*/
void scan_inclusive(thrust::device_vector<int>& vec_d){
	thrust::inclusive_scan(vec_d.begin(), vec_d.end(), vec_d.begin());
}

void scan_exclusive(thrust::device_vector<int>& vec_d){
	thrust::exclusive_scan(vec_d.begin(), vec_d.end(), vec_d.begin());
}

void scan_inclusive(thrust::device_vector<int>& vec_d, int data_num){
	if( data_num >= vec_d.size() )
		thrust::inclusive_scan(vec_d.begin(), vec_d.end(), vec_d.begin());
	else
		thrust::inclusive_scan(vec_d.begin(), vec_d.begin()+data_num, vec_d.begin());
}

void scan_exclusive(thrust::device_vector<int>& vec_d, int data_num){
	if( data_num >= vec_d.size() )
		thrust::exclusive_scan(vec_d.begin(), vec_d.end(), vec_d.begin());
	else
		thrust::exclusive_scan(vec_d.begin(), vec_d.begin()+data_num, vec_d.begin());
}

extern "C" 
void scan_inclusiveull(thrust::device_vector<unsigned long long>& vec_d){
	thrust::inclusive_scan(vec_d.begin(), vec_d.end(), vec_d.begin());
}

extern "C" 
void scan_exclusiveull(thrust::device_vector<unsigned long long>& vec_d){
	thrust::exclusive_scan(vec_d.begin(), vec_d.end(), vec_d.begin());
}

