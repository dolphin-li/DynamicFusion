/***********************************************************/
/**	\file
	\brief		cuda utils used by Kinect Fusion
	\details	
	\author		Yizhong Zhang
	\date		11/13/2013
*/
/***********************************************************/
#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime_api.h"
#pragma comment(lib, "cudart.lib")


#define cudaSafeCall ___cudaSafeCall


void ___cudaSafeCall(cudaError_t err, const char* msg = NULL);

static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }



#endif