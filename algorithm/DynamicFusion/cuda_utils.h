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


static inline void ___cudaSafeCall(cudaError_t err, char* msg=NULL)
{
	//// debug
	//err = cudaThreadSynchronize();
	//if (cudaSuccess != err){
	//	printf("CUDA error1(%s): %s\n", msg, cudaGetErrorString(err));
	//	exit(-1);
	//}
	//// end debug

	if (cudaSuccess != err){
		printf( "CUDA error(%s): %s\n", msg, cudaGetErrorString(err) );
		exit(-1);
	}
}        

static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }



#endif