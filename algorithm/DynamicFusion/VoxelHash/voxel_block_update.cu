/***********************************************************/
/**	\file
	\brief		update voxel block (integrate)
	\details	
	\author		Yizhong Zhang
	\date		12/9/2013
*/
/***********************************************************/
#include "device_utils.h"
#include "voxel_hashing_device.h"
#include "voxel_hashing_internal.h"
#include "voxel_block_hash_table.cuh"
#include "kernel_containers.h"

//	==================================================================
template<int CTA_SIZE_, typename T>
static __device__ __forceinline__ void reduceMin(volatile T* buffer)
{
	int tid = dfusion::Block::flattenedThreadId();
	T val =  buffer[tid];

	if (CTA_SIZE_ >= 1024) { if (tid < 512) buffer[tid] = val = min(val, buffer[tid + 512]); __syncthreads(); }
	if (CTA_SIZE_ >=  512) { if (tid < 256) buffer[tid] = val = min(val, buffer[tid + 256]); __syncthreads(); }
	if (CTA_SIZE_ >=  256) { if (tid < 128) buffer[tid] = val = min(val, buffer[tid + 128]); __syncthreads(); }
	if (CTA_SIZE_ >=  128) { if (tid <  64) buffer[tid] = val = min(val, buffer[tid +  64]); __syncthreads(); }

	if (tid < 32){
		if (CTA_SIZE_ >=   64) { buffer[tid] = val = min(val, buffer[tid +  32]); }
		if (CTA_SIZE_ >=   32) { buffer[tid] = val = min(val, buffer[tid +  16]); }
		if (CTA_SIZE_ >=   16) { buffer[tid] = val = min(val, buffer[tid +   8]); }
		if (CTA_SIZE_ >=    8) { buffer[tid] = val = min(val, buffer[tid +   4]); }
		if (CTA_SIZE_ >=    4) { buffer[tid] = val = min(val, buffer[tid +   2]); }
		if (CTA_SIZE_ >=    2) { buffer[tid] = val = min(val, buffer[tid +   1]); }
	}
}

template<int CTA_SIZE_, typename T>
static __device__ __forceinline__ void reduceMax(volatile T* buffer)
{
	int tid = dfusion::Block::flattenedThreadId();
	T val =  buffer[tid];

	if (CTA_SIZE_ >= 1024) { if (tid < 512) buffer[tid] = val = max(val, buffer[tid + 512]); __syncthreads(); }
	if (CTA_SIZE_ >=  512) { if (tid < 256) buffer[tid] = val = max(val, buffer[tid + 256]); __syncthreads(); }
	if (CTA_SIZE_ >=  256) { if (tid < 128) buffer[tid] = val = max(val, buffer[tid + 128]); __syncthreads(); }
	if (CTA_SIZE_ >=  128) { if (tid <  64) buffer[tid] = val = max(val, buffer[tid +  64]); __syncthreads(); }

	if (tid < 32){
		if (CTA_SIZE_ >=   64) { buffer[tid] = val = max(val, buffer[tid +  32]); }
		if (CTA_SIZE_ >=   32) { buffer[tid] = val = max(val, buffer[tid +  16]); }
		if (CTA_SIZE_ >=   16) { buffer[tid] = val = max(val, buffer[tid +   8]); }
		if (CTA_SIZE_ >=    8) { buffer[tid] = val = max(val, buffer[tid +   4]); }
		if (CTA_SIZE_ >=    4) { buffer[tid] = val = max(val, buffer[tid +   2]); }
		if (CTA_SIZE_ >=    2) { buffer[tid] = val = max(val, buffer[tid +   1]); }
	}
}

//	==================================================================

__global__ void
createScaleDepth (const PtrStepSz<float> depth, PtrStep<float> scaled, const dfusion::Intr intr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= depth.cols || y >= depth.rows)
		return;

	float Dp = depth.ptr (y)[x];

	float xl = (x - intr.cx) / intr.fx;
	float yl = (y - intr.cy) / intr.fy;
	float lambda = sqrtf (xl * xl + yl * yl + 1);

	scaled.ptr (y)[x] = Dp * lambda; 
}

//	==================================================================

struct VoxelBlockUpdater{
	enum{
		CTA_SIZE = BLOCK_DIM * BLOCK_DIM * BLOCK_DIM
	};
	
	PtrStepSz<float>			depth;			//	the scaled depth
	dfusion::Intr				intr;			//	intrinsic parameters of camera

	dfusion::Mat33				Rw2c;			//	world to camera
	float3						tw2c;

	float				block_size;		//	edge length of the block cube, voxel_size * BLOCK_DIM
	float				voxel_size;		//	edge length of the voxel cube
	float				trunc_dist;		//	truncation distance
	unsigned char		max_weight;		//	max weight

	PtrSz<HashEntry>				visible_hash_entry;	//	the hash table, organized in sequence of each entry
	mutable PtrSz<VoxelBlock>		voxel_block;		//	the array of voxel block

	mutable PtrSz<unsigned char>	delete_hash_entry;	//	label whether this hash entry should be deleted
	float							abs_tsdf_thre;		//	if minimun tsdf in this block is bigger than this threshold, delete this entry

	__device__ __forceinline__ void operator () () const {
		//	calculate coordinate of this voxel, one thread each voxel
		int voxel_x = threadIdx.x;
		int voxel_y = threadIdx.y;
		int voxel_z = threadIdx.z;
		int voxel_idx = (voxel_z * BLOCK_DIM + voxel_y) * BLOCK_DIM + voxel_x;

		//	coordinate of the voxel block, one CUDA block each voxel block
		int block_x = visible_hash_entry[blockIdx.x].position[0];
		int block_y = visible_hash_entry[blockIdx.x].position[1];
		int block_z = visible_hash_entry[blockIdx.x].position[2];
		int block_idx = visible_hash_entry[blockIdx.x].pointer;

		//	copy tsdf and weight to shared memory
		__shared__ float tsdf[CTA_SIZE];
		__shared__ unsigned char weight[CTA_SIZE];
		tsdf[voxel_idx]		= voxel_block[block_idx].voxel[voxel_idx].sdf;
		weight[voxel_idx]	= voxel_block[block_idx].voxel[voxel_idx].weight;
		__syncthreads ();

		//	world coordinate
		float3 xyz = make_float3(
			block_x * block_size + (voxel_x + 0.5f) * voxel_size,
			block_y * block_size + (voxel_y + 0.5f) * voxel_size,
			block_z * block_size + (voxel_z + 0.5f) * voxel_size );

		//	transform to camera coordinate
		xyz = Rw2c * xyz + tw2c;

		//	project the point onto screen
		float3 uvd = intr.xyz2uvd(xyz);
		int2 ukr;
		ukr.x = __float2int_rn (uvd.x);
		ukr.y = __float2int_rn (uvd.y);

		//	if this voxel is in the view frustum
		if (ukr.x >= 0 && ukr.y >= 0 && ukr.x < depth.cols && ukr.y < depth.rows){
			//	calculate signed distance function
			float depthVal = depth(ukr.y, ukr.x) * 0.001f;
			float3 dxyz = intr.uvd2xyz(make_float3(ukr.x, ukr.y, depthVal));
			float sdf = xyz.z - dxyz.z;

			//	if the projection point has depth value and this voxel is able to update
			if (depthVal > 0.001f && sdf >= -trunc_dist)		//	meters
			{
				float _tsdf = min (1.0f, sdf / trunc_dist);		//	range -1 to +1, negative means behind observed depth

				float	tsdf_prev	= tsdf[voxel_idx];
				int		weight_prev = weight[voxel_idx];

				//int Wrk = xyz.z>2.5f ? 1.0f : (3.0f - xyz.z)/0.5f;
				int Wrk = (3.5f - xyz.z)/0.5f;

				if( Wrk > 0 ){
					float			tsdf_new	= (tsdf_prev * weight_prev + Wrk * _tsdf) / (weight_prev + Wrk);
					unsigned char	weight_new	= min (weight_prev + Wrk, max_weight);

					tsdf[voxel_idx]		= tsdf_new;
					weight[voxel_idx]	= weight_new;
				}
			}
		}

		//	write tsdf and weight to voxel block
		voxel_block[block_idx].voxel[voxel_idx].sdf		= tsdf[voxel_idx];
		voxel_block[block_idx].voxel[voxel_idx].weight	= weight[voxel_idx];

		//	calculate max weight and min abs tsdf
		tsdf[voxel_idx] = fabsf( tsdf[voxel_idx] );

		__syncthreads ();
		reduceMin<CTA_SIZE>(tsdf);
		reduceMax<CTA_SIZE>(weight);
		
		//	in the first thread, check whether this block should be deleted
		if( voxel_idx == 0 ){
			if( weight[0] == 0 || tsdf[0] > abs_tsdf_thre )
				delete_hash_entry[blockIdx.x] = 1;
			else
				delete_hash_entry[blockIdx.x] = 0;
		}
	}


};

__global__ void updateVoxelBlockKernel( const VoxelBlockUpdater updater ){
	updater();
}

//	==================================================================
struct VoxelBlockDeleter : public VoxelBlockHashTable{
	int							visible_hash_entry_number;
	PtrSz<HashEntry>			visible_hash_entry;	
	PtrSz<unsigned char>		delete_hash_entry;

	__device__ __forceinline__ void operator() () const{
		int idx = threadIdx.x + blockIdx.x * blockDim.x;

		if( idx >= visible_hash_entry_number )
			return;

		if( delete_hash_entry[idx] ){
			int X = visible_hash_entry[idx].position[0];
			int Y = visible_hash_entry[idx].position[1];
			int Z = visible_hash_entry[idx].position[2];
			DeleteHashEntryStaggered(X, Y, Z);
		} 
	}

};

__global__ void deleteVoxelBlockKernel( const VoxelBlockDeleter deleter ){
	deleter();
}

//	==================================================================
void updateVoxelBlock(
	const PtrStepSz<float>&			depth, 
	const dfusion::Intr&			intr,
	const dfusion::Mat33&			Rw2c,
	const float3&					tw2c,
	float							block_size, 
	float							voxel_size,
	float							trunc_dist,
	DeviceArray<HashEntry>&			hash_entry, 
	int								bucket_size,
	DeviceArray<unsigned int>&		hash_bucket_atomic_lock, 
	DeviceArray<VoxelBlock>&		voxel_block,
	DeviceArray<int>&				available_voxel_block, 
	DeviceArray<int>&				hash_parameters,
	const DeviceArray<HashEntry>&	visible_hash_entry,
	int								visible_hash_entry_number,
	DeviceArray<unsigned char>		delete_hash_entry,
	float							abs_tsdf_thre )
{
	if (visible_hash_entry_number == 0)
		return;
	//	update each voxel in voxel block
	VoxelBlockUpdater updater;
	updater.depth			= depth;
	updater.intr			= intr;

	updater.Rw2c			= Rw2c;
	updater.tw2c			= tw2c;

	updater.block_size		= block_size;
	updater.voxel_size		= voxel_size;
	updater.trunc_dist		= trunc_dist;
	updater.max_weight		= 128;

	updater.visible_hash_entry	= visible_hash_entry;
	updater.voxel_block			= voxel_block;

	updater.delete_hash_entry	= delete_hash_entry;
	updater.abs_tsdf_thre		= abs_tsdf_thre;

	dim3 block (BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
	dim3 grid (visible_hash_entry_number);

	updateVoxelBlockKernel<<<grid, block>>>( updater );

	//	setup block bucket atomic lock
	int threadPerBlock = 256;
	int blocksPerGrid = divUp(hash_bucket_atomic_lock.size(), threadPerBlock);

	initHashBucketAtomicLock << <blocksPerGrid, threadPerBlock >> >(hash_bucket_atomic_lock);
	cudaSafeCall(cudaGetLastError(), "updateVoxelBlock::initHashBucketAtomicLock");

	//	delete hash entries that has been marked to delete
	VoxelBlockDeleter deleter;

	deleter.hash_table_size			= hash_entry.size() / bucket_size;
	deleter.bucket_size				= bucket_size;
	deleter.hash_entry				= hash_entry;
	deleter.hash_bucket_atomic_lock	= hash_bucket_atomic_lock;
	deleter.available_voxel_block	= available_voxel_block;
	deleter.hash_parameters			= hash_parameters;

	deleter.visible_hash_entry_number	= visible_hash_entry_number;
	deleter.visible_hash_entry			= visible_hash_entry;
	deleter.delete_hash_entry			= delete_hash_entry;

	threadPerBlock = 256;
	blocksPerGrid = divUp(visible_hash_entry_number, threadPerBlock);

	deleteVoxelBlockKernel<<<blocksPerGrid, threadPerBlock>>>( deleter );
	cudaSafeCall(cudaGetLastError(), "updateVoxelBlock::deleteVoxelBlockKernel");
}


