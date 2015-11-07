/***********************************************************/
/**	\file
	\brief		voxel hashing device functions
	\details	
	\author		Yizhong Zhang
	\date		12/7/2013
*/
/***********************************************************/
#ifndef	__VOXEL_HASHING_DEVICE_H__
#define __VOXEL_HASHING_DEVICE_H__

#include "voxel_hashing_internal.h"

/**
	hashing functions
*/
__host__ __device__ __forceinline__ int
	hashing_func(int x, int y, int z, int n)
{
	const int p1 = 73856093;
	const int p2 = 19349669;
	const int p3 = 83492791;
	//return unsigned int( (x * p1) ^ (y * p2) ^ (z * p3) ) % n;
	return unsigned int( ((x+9973) * p1) ^ ((y+19997) * p2) ^ ((z+29989) * p3) ) % n;
}

__host__ __device__ __forceinline__ int
	hashing_func(int3 xyz, int n)
{
	return hashing_func(xyz.x, xyz.y, xyz.z, n);
}


/**
	transform world coordinate x y z to block coordinate i j k
*/
__device__ __forceinline__ void
	coord2block(int& i, int& j, int& k, float x, float y, float z, float block_size)
{
	i = __float2int_rd(x/block_size);
	j = __float2int_rd(y/block_size);
	k = __float2int_rd(z/block_size);
}

__device__ __forceinline__ int3 
	coord2block(float x, float y, float z, float block_size)
{
	int i, j, k;
	coord2block(i, j, k, x, y, z, block_size);
	return make_int3(i, j, k);
}

__device__ __forceinline__ int3 
	coord2block(float3 xyz, float block_size)
{
	return coord2block(xyz.x, xyz.y, xyz.z, block_size);
}

/**
	transform world coordinate x y z to voxel coordinate i j k
*/
__device__ __forceinline__ void
	coord2voxel(int& i, int& j, int& k, float x, float y, float z, float voxel_size)
{
	i = __float2int_rd(x/voxel_size);
	j = __float2int_rd(y/voxel_size);
	k = __float2int_rd(z/voxel_size);
}

__device__ __forceinline__ int3 
	coord2voxel(float x, float y, float z, float voxel_size)
{
	int i, j, k;
	coord2voxel(i, j, k, x, y, z, voxel_size);
	return make_int3(i, j, k);
}

__device__ __forceinline__ int3 
	coord2voxel(float3 xyz, float voxel_size)
{
	return coord2voxel(xyz.x, xyz.y, xyz.z, voxel_size);
}


#endif