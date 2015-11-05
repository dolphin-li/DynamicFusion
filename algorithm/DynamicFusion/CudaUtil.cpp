#include "cuda_utils.h"
#include <stdio.h>
void ___cudaSafeCall(cudaError_t err, const char* msg)
{
	//// debug
	//err = cudaThreadSynchronize();
	//if (cudaSuccess != err){
	//	printf("CUDA error1(%s): %s\n", msg, cudaGetErrorString(err));
	//	exit(-1);
	//}
	//// end debug

	if (cudaSuccess != err){
		printf("CUDA error(%s): %s\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}