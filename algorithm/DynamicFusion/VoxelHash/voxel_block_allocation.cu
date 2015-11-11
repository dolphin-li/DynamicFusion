/***********************************************************/
/**	\file
	\brief		alloc voxel blocks within truncation distance of depth pixel
	\details	
	\author		Yizhong Zhang
	\date		12/7/2013
*/
/***********************************************************/
#include "voxel_hashing_device.h"
#include "voxel_hashing_internal.h"
#include "voxel_block_hash_table.cuh"
#include <helper_math.h>
#include "device_utils.h"

#define GPRINT(a) {cudaSafeCall(cudaThreadSynchronize(), #a);printf("%s\n",#a);}

//	==================================================================
__global__ void initHashBucketAtomicLock(PtrSz<unsigned int> hash_bucket_atomic_lock){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if( idx < hash_bucket_atomic_lock.size )
		hash_bucket_atomic_lock[idx] = 0;
}

//	==================================================================
struct VoxelBlockAllocator : public VoxelBlockHashTable{
	enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8 };

	PtrStepSz<float>	depth;	//	the input depth map
	dfusion::Intr		intr;	//	intrinsic parameters of camera
	int		cols, rows;			//	resolution of depth image

	dfusion::Mat33		Rc2w;	//	camera to world
	float3	tc2w;

	float	trunc_dist;			//	truncation distance

	__device__ __forceinline__ float3 get_point_in_camera_coord(int x, int y, float d) const {
		return intr.uvd2xyz(x, y, d);
	}
	
	__device__ __forceinline__ void operator () () const {
		int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
		int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

		if (x >= cols || y >= rows)
			return;

		float d = depth.ptr (y)[x];
		if( d < 0.3f )	//	input depth < 0.3m, illegal depth
			return;

		//	the following code traverses each voxel block that the line ray intersect near depth
		//	algorithm comes from the paper: A fast voxel traversal algorithm for ray tracing
		float3 ray_origin		= tc2w;
		float3 ray_end			= Rc2w * get_point_in_camera_coord (x, y, d) + tc2w;
		float3 ray_direction	= normalize (ray_end - ray_origin);

		ray_origin		= ray_end - ray_direction * trunc_dist;
		ray_end			= ray_end + ray_direction * trunc_dist;
		ray_direction	= ray_end - ray_origin;

		//	the following code traverses the voxels between ray_origin and ray_end
		int X, Y, Z;
		coord2block(X, Y, Z, ray_origin.x, ray_origin.y, ray_origin.z, block_size);

		int stepX, stepY, stepZ;
		stepX = sgn(ray_direction.x);
		stepY = sgn(ray_direction.y);
		stepZ = sgn(ray_direction.z);

		float tMaxX, tMaxY, tMaxZ;
		tMaxX = ( block_size*(X + (stepX+1)/2) - ray_origin.x ) / ray_direction.x;
		tMaxY = ( block_size*(Y + (stepY+1)/2) - ray_origin.y ) / ray_direction.y;
		tMaxZ = ( block_size*(Z + (stepZ+1)/2) - ray_origin.z ) / ray_direction.z;
		if( isnan(tMaxX) || isinf(tMaxX) ) tMaxX = 1e6f;
		if( isnan(tMaxY) || isinf(tMaxY) ) tMaxY = 1e6f;
		if( isnan(tMaxZ) || isinf(tMaxZ) ) tMaxZ = 1e6f;

		float tDeltaX, tDeltaY, tDeltaZ;
		tDeltaX = stepX * block_size / ray_direction.x;
		tDeltaY = stepY * block_size / ray_direction.y;
		tDeltaZ = stepZ * block_size / ray_direction.z;
		if( isnan(tDeltaX) || isinf(tDeltaX) ) tDeltaX = 1e6f;
		if( isnan(tDeltaY) || isinf(tDeltaY) ) tDeltaY = 1e6f;
		if( isnan(tDeltaZ) || isinf(tDeltaZ) ) tDeltaZ = 1e6f;

		int count = 0;
		while(count<50){//	infinite loop guard
			count ++;
			//	--------	inside this code segment, X Y Z is the cell we want to process	--------

			//	we insert an entry into the hash table, but it is not guaranteed to insert successfully
			//	but it doesn't matter even if insert failed this time.
			int entry_id;
			InsertHashEntryStaggered(entry_id, X, Y, Z);
		
			//	------------------------------------------------------------------------------------

			if( tMaxX > 1.0f && tMaxY > 1.0f && tMaxZ > 1.0f )
				break;

			if( tMaxX < tMaxY && tMaxX < tMaxZ ){
				X += stepX;
				tMaxX += tDeltaX;
			}
			else if(tMaxY < tMaxZ){
				Y += stepY;
				tMaxY += tDeltaY;
			}
			else{
				Z += stepZ;
				tMaxZ += tDeltaZ;
			}
		}

	}

};

__global__ void voxelBlockAllocKernel (const VoxelBlockAllocator allocator) {
	allocator ();
}

//	==================================================================
__global__ void resetVoxelBlock(
	PtrSz<VoxelBlock>	voxel_block, 
	PtrSz<int>			available_voxel_block,
	int					start_id )
{
	int voxel_id = threadIdx.x;
	int block_id = available_voxel_block[start_id + blockIdx.x];

	voxel_block[block_id].voxel[voxel_id].sdf			= 0.0f;
	voxel_block[block_id].voxel[voxel_id].colorRGB[0]	= 0;
	voxel_block[block_id].voxel[voxel_id].colorRGB[1]	= 0;
	voxel_block[block_id].voxel[voxel_id].colorRGB[2]	= 0;
	voxel_block[block_id].voxel[voxel_id].weight		= 0;
}


//	==================================================================
void allocVoxelBlock(
	const PtrStepSz<float>&		depth, 
	const dfusion::Intr&					intr, 
	const dfusion::Mat33&				Rc2w,
	const float3&				tc2w,
	float						block_size, 
	float						trunc_dist,
	DeviceArray<HashEntry>&		hash_entry, 
	int							bucket_size,
	DeviceArray<unsigned int>&	hash_bucket_atomic_lock, 
	DeviceArray<VoxelBlock>&	voxel_block,
	DeviceArray<int>&			available_voxel_block, 
	DeviceArray<int>&			hash_parameters,
	int&						voxel_block_number,
	int3						chunk_dim,
	float3						chunk_min_xyz,
	float						chunk_size,
	DeviceArray<unsigned char>&	chunk_on_CPU )
{
	//	setup block bucket atomic lock
	int threadPerBlock = 256;
	int blocksPerGrid = divUp(hash_bucket_atomic_lock.size(), threadPerBlock);

	initHashBucketAtomicLock<<<blocksPerGrid, threadPerBlock>>>(hash_bucket_atomic_lock);
	cudaSafeCall(cudaGetLastError(), "allocVoxelBlock::initHashBucketAtomicLock");
	GPRINT(1);
	//	initial voxel block number
	std::vector<int> param;
	param.resize( hash_parameters.size() );
	hash_parameters.download(param.data());
	voxel_block_number = param[0];
	GPRINT(2);

	//	alloc block
	VoxelBlockAllocator allocator;

	allocator.hash_table_size			= hash_entry.size() / bucket_size;
	allocator.bucket_size				= bucket_size;
	allocator.hash_entry				= hash_entry;
	allocator.hash_bucket_atomic_lock	= hash_bucket_atomic_lock;
	allocator.available_voxel_block		= available_voxel_block;
	allocator.hash_parameters			= hash_parameters;

	allocator.depth	= depth;
	allocator.intr = intr;
	allocator.cols = depth.cols;
	allocator.rows = depth.rows;

	allocator.Rc2w = Rc2w;
	allocator.tc2w = tc2w;

	allocator.block_size = block_size;
	allocator.trunc_dist = trunc_dist;

	allocator.chunk_dim		= chunk_dim;
	allocator.chunk_min_xyz	= chunk_min_xyz;
	allocator.chunk_size	= chunk_size;
	allocator.chunk_on_CPU	= chunk_on_CPU;

	dim3 block (32, 8);
	dim3 grid (divUp (depth.cols, block.x), divUp (depth.rows, block.y));

	voxelBlockAllocKernel << <grid, block >> >(allocator);
	cudaSafeCall(cudaGetLastError(), "allocVoxelBlock::voxelBlockAllocKernel");

	GPRINT(3);
	//	clear the data of new allocated voxel blocks
	//	in the previous function, only allocation is performed for hash table
	//	so the new allocated voxel blocks are from voxel_block_number to new voxel_block_number
	param.resize( hash_parameters.size() );
	hash_parameters.download(param.data());
	GPRINT(4);
	int start_id	= voxel_block_number;
	int end_id		= param[0];
	if( end_id > start_id ){
		int threadPerBlock = BLOCK_DIM*BLOCK_DIM*BLOCK_DIM;
		int blocksPerGrid = end_id - start_id;
		resetVoxelBlock<<<blocksPerGrid, threadPerBlock>>>(
			voxel_block, available_voxel_block, start_id);
		cudaSafeCall(cudaGetLastError(), "allocVoxelBlock::resetVoxelBlock");

		voxel_block_number = end_id;
	}


}



