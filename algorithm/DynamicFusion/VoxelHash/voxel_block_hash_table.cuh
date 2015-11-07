/***********************************************************/
/**	\file
	\brief		the base class of voxel block hash table
	\details	
	\author		Yizhong Zhang
	\date		12/8/2013
*/
/***********************************************************/
#ifndef __VOXEL_BLOCK_HASH_TABLE_H__
#define __VOXEL_BLOCK_HASH_TABLE_H__

#include "voxel_hashing_device.h"
#include "voxel_hashing_internal.h"
#include <helper_math.h>
/**
	the base class of voxel block hash table

	if hash table should be used, derive from this struct
*/
struct VoxelBlockHashTable{
	int		hash_table_size;		//	size of the hash table, should be identical to hash_entry.size/bucket_size
	int		bucket_size;			//	bucket size of the hash table

	mutable PtrSz<HashEntry>		hash_entry;					//	the hash table, organized in sequence of each entry
	mutable PtrSz<unsigned int>		hash_bucket_atomic_lock;	//	lock for writting of each bucket
	mutable PtrSz<int>				available_voxel_block;		//	heap of available voxels blocks
	mutable PtrSz<int>				hash_parameters;			//	parameters, hash_parameters[0] is the first available block id

	//	variables of chunk
	int3	chunk_dim;				//	dimension of chunks
	float3	chunk_min_xyz;			//	min coordinate of chunks
	float	chunk_size;				//	size of each chunk
	float	block_size;				//	the size of each block

	mutable PtrSz<unsigned char>	chunk_on_CPU;				//	label of whether this chunk is on CPU, if so, we cannot alloc new blocks

	__device__ __forceinline__ int sgn(float val) const {
		return (0.0f < val) - (val < 0.0f);
	}

	__device__ __forceinline__ int mod(int a, int b) const {
		return (a%b + b) % b;
	}

	__device__ __forceinline__ int div_rd(int a_numerator, int a_denominator) const{
		return (a_numerator / a_denominator) + ((a_numerator % a_denominator) >> 31);
		//return (a_numerator + a_denominator) / a_denominator - 1;
	}

	__device__ __forceinline__ int GetChunkIdx(int X, int Y, int Z) const{
		float3	block_xyz	= make_float3(X*block_size+1e-4f, Y*block_size+1e-4f, Z*block_size+1e-4f);
		float3	r = block_xyz - chunk_min_xyz;
		int3	chunk_grid	= make_int3(r.x/chunk_size, r.y/chunk_size, r.z/chunk_size );

		return (chunk_grid.z * chunk_dim.y + chunk_grid.y) * chunk_dim.x + chunk_grid.x;
	}

	__device__ __forceinline__ int GetOffsetHashEntryId(int curr_id, short offset) const{
		int target_id = curr_id + offset;
		return mod(target_id, hash_entry.size);
	}

	__device__ __forceinline__ int GetNextHashEntryId(int curr_id) const{
		int target_id = curr_id + 1;
		return mod(target_id, hash_entry.size);
	}

	__device__ __forceinline__ int GetNextNonBucketEndHashEntryId(int curr_id) const{
		int target_id = curr_id + 1;
		if( (target_id+1) % bucket_size == 0 )
			target_id ++;
		return mod(target_id, hash_entry.size);
	}

	__device__ __forceinline__ short GetOffset(int hash_list_end, int insert_entry_id) const{
		const int half_hash_entry_size = hash_entry.size / 2;

		int offset = insert_entry_id - hash_list_end;

		if( offset > half_hash_entry_size )
			offset -= hash_entry.size;
		else if( offset < -half_hash_entry_size )
			offset += hash_entry.size;

		return offset;
	}

	__device__ __forceinline__ bool IsChunkOnCPU(int X, int Y, int Z) const{
		int		chunk_idx = GetChunkIdx(X, Y, Z);
		return chunk_on_CPU[chunk_idx] ? true : false;
	}

	__device__ __forceinline__ int GetHashEntryId(int X, int Y, int Z, 
		int& bucket_id, int& bucket_start, int& bucket_end, int& hash_list_end) const
	{
		bucket_id		= hashing_func(X, Y, Z, hash_table_size);
		bucket_start	= bucket_id * bucket_size;
		bucket_end		= bucket_start + bucket_size - 1;

		//	search in the bucket to see whether X Y Z already exist
		for( int search_id = bucket_start; search_id <= bucket_end; search_id ++ ){
			if( hash_entry[search_id].pointer >= 0 && 
				hash_entry[search_id].position[0] == X &&
				hash_entry[search_id].position[1] == Y &&
				hash_entry[search_id].position[2] == Z )
			{
				return search_id;	//	found the hash entry
			}
		}

		//	if didn't found within this bucket, search in the linked list
		const int count_guard = 1000;	//	the maximum length of the linked list
		hash_list_end	= bucket_end;	//	the last entry before searched
		if( hash_entry[hash_list_end].offset != 0 ){
			int count = 0;
			int search_id = GetOffsetHashEntryId(hash_list_end, hash_entry[hash_list_end].offset);
			do{
				if( hash_entry[search_id].position[0] == X &&
					hash_entry[search_id].position[1] == Y &&
					hash_entry[search_id].position[2] == Z )
				{
					return search_id;
				}
				else{
					hash_list_end = search_id;
					search_id = GetOffsetHashEntryId(hash_list_end, hash_entry[hash_list_end].offset);
				}
				count ++;
			}while(hash_list_end != search_id && count < count_guard);

			if( count == count_guard ){
				printf("the linked list is extremly long, maybe caused by bugs");
			}
		}

		return -1;
	}

	__device__ __forceinline__ int GetHashEntryId(int3 block_position,
		int& bucket_id, int& bucket_start, int& bucket_end, int& hash_list_end) const
	{
		return GetHashEntryId(block_position.x, block_position.y, block_position.z,
			bucket_id, bucket_start, bucket_end, hash_list_end);
	}

	__device__ __forceinline__ int GetHashEntryId(int3 block_position) const
	{
		int bucket_id, bucket_start, bucket_end, hash_list_end;
		return GetHashEntryId(block_position, bucket_id, bucket_start, bucket_end, hash_list_end);
	}

	__device__ __forceinline__ int GetBucketId(int entry_id) const{
		return entry_id / bucket_size;
	}

	__device__ __forceinline__ int LockBucket(int bucket_id) const{
		int old_lock = atomicCAS( &hash_bucket_atomic_lock[bucket_id], 0, 1 );
		return !old_lock;	//	if originally not locked, it is locked now
	}

	__device__ __forceinline__ void UnlockBucket(int bucket_id) const{
		hash_bucket_atomic_lock[bucket_id] = 0;		//	it is free to unlock any time without atomic operation
	}

	__device__ __forceinline__ int SetHashEntryAndAllocVoxelBlock(int entry_id, int offset, int X, int Y, int Z) const{
		//	calling this function assumes that we have got the correct entry 
		//	and we have already locked this bucket

		hash_entry[entry_id].offset = 0;
		hash_entry[entry_id].position[0] = X;
		hash_entry[entry_id].position[1] = Y;
		hash_entry[entry_id].position[2] = Z;

		//	alloc voxel block memory, the first available block is 
		int list_pos = atomicAdd( &hash_parameters[0], 1 );

		//	if no more voxel block is able to be allocated, just stop here
		if( list_pos >= available_voxel_block.size ){
			hash_parameters[0] = available_voxel_block.size;
			return 0;
		}

		hash_entry[entry_id].pointer = available_voxel_block[list_pos];

		return 1;
	}

	__device__ __forceinline__ int ClearHashEntryAndFreeVoxelBlock(int entry_id) const{
		//	calling this function assumes that we have got the correct entry 
		//	and it is safe to clear

		//if( hash_entry[entry_id].offset != 0 ){
		//	printf("deleted: %d, %d : %d, %d, %d --- %d\n", 
		//		entry_id, 
		//		hash_entry[entry_id].offset,
		//		hash_entry[entry_id].position[0],
		//		hash_entry[entry_id].position[1],
		//		hash_entry[entry_id].position[2],
		//		hash_entry[entry_id].pointer
		//		);
		//}


		int pointer = hash_entry[entry_id].pointer;

		hash_entry[entry_id].pointer = -1;

		//	free voxel block memory
		int list_pos = atomicSub( &hash_parameters[0], 1 ) - 1;
		available_voxel_block[list_pos] = pointer;

		return 1;
	}

	__device__ __forceinline__ int InsertHashEntryStaggered(int& entry_id, int X, int Y, int Z) const{
		//	return value
		//	1:	insert successful, entry_id is written
		//	0:	this voxel already in hash table, don't need to insert, entry_id is written
		//	-1:	this bucket has been locked, insert failed
		//	-2:	empty entry cannot be found, insert failed
		//	-3:	this chunk is on CPU, insert failed
		//	-4:	no more voxel block can be allocated

		if( IsChunkOnCPU(X, Y, Z) )
			return -3;

		int bucket_id, bucket_start, bucket_end, hash_list_end;
		entry_id = GetHashEntryId(X, Y, Z, bucket_id, bucket_start, bucket_end, hash_list_end);

		//	if this block is already allocated, return 0, indicating this entry already exist
		if( entry_id >= 0 )	
			return 0;

		//	if this block is not allocated yet, search an empty space in the bucket
		int insert_entry_id = bucket_start;
		for(; insert_entry_id<=bucket_end; insert_entry_id++){
			if( hash_entry[insert_entry_id].pointer < 0 ){
				//	this bucket has been locked for writing, don't insert this time
				if( !LockBucket(bucket_id) )
					return -1;

				entry_id = insert_entry_id;
				return SetHashEntryAndAllocVoxelBlock(insert_entry_id, 0, X, Y, Z);
			}
		}

		//	the bucket is full, search after the bucket until an empty entry, and insert into the back of the linked list
		insert_entry_id = GetNextNonBucketEndHashEntryId(bucket_end);
		const int count_guard = 1000;	//	the max distance to search
		int count = 0;
		while(hash_entry[insert_entry_id].pointer >= 0 && count < count_guard){
			insert_entry_id = GetNextNonBucketEndHashEntryId(insert_entry_id);
			count ++;
		}

		//	we didn't find an empty entry
		if( count == count_guard )
			return -2;

		//	we have found the entry, lock the bucket and insert now
		int insert_bucket_id = GetBucketId(insert_entry_id);
		if( !LockBucket(insert_bucket_id) || !LockBucket(bucket_id) )
			return -1;
		if( !SetHashEntryAndAllocVoxelBlock(insert_entry_id, 0, X, Y, Z) )
			return -4;

		hash_entry[hash_list_end].offset = GetOffset(hash_list_end, insert_entry_id);
		entry_id = insert_entry_id;

		return 1;
	}

	__device__ __forceinline__ int DeleteHashEntryStaggered(int X, int Y, int Z) const{
		//	return value
		//	1:	delete successful
		//	0:	cannot find this hash entry
		//	-1:	this bucket has been locked, delete failed

		int bucket_id, bucket_start, bucket_end, hash_list_end;
		int entry_id = GetHashEntryId(X, Y, Z, bucket_id, bucket_start, bucket_end, hash_list_end);

		//	if this block is doesn't exist
		if( entry_id < 0 )	
			return 0;

		//	find the bucket that this entry lies
		int delete_bucket_id = GetBucketId(entry_id);

		//	if this entry lies in its own bucket
		if( delete_bucket_id == bucket_id ){
			//	if this entry is a head of a linked list, copy the next element and delete it
			if( ( entry_id - 1 == bucket_id * bucket_size ) && ( hash_entry[entry_id].offset != 0 ) ){
				//	we have to lock this bucket and the bucket containing the next entry
				int next_entry_id = GetOffsetHashEntryId( entry_id, hash_entry[entry_id].offset );
				int next_bucket_id = GetBucketId( next_entry_id );
				if( !LockBucket(delete_bucket_id) || !LockBucket(next_bucket_id) )
					return -1;

				//	we have successfully locked the bucket, now delete the entry
				int new_offset = hash_entry[next_entry_id].offset ? hash_entry[entry_id].offset + hash_entry[next_entry_id].offset : 0;
				int ret = ClearHashEntryAndFreeVoxelBlock(entry_id);
				hash_entry[entry_id] = hash_entry[next_entry_id];
				hash_entry[entry_id].offset = new_offset;
				hash_entry[next_entry_id].pointer = -1;
				return ret;
			}
			//	else delete directly, no lock is needed since it doesn't affect any others
			else{
				return ClearHashEntryAndFreeVoxelBlock(entry_id);
			}
		}
		//	this entry is one element inside the linked list
		else{
			int prev_bucket_id = GetBucketId( hash_list_end );
			if( !LockBucket(delete_bucket_id) || !LockBucket(prev_bucket_id) )
				return -1;

			//	we have successfully locked the bucket, now delete the entry
			int new_offset = hash_entry[entry_id].offset ? hash_entry[hash_list_end].offset + hash_entry[entry_id].offset : 0;
			hash_entry[hash_list_end].offset = new_offset;
			return ClearHashEntryAndFreeVoxelBlock(entry_id);
		}
	}

	__device__ __forceinline__ int InsertHashEntry(int& entry_id, int X, int Y, int Z) const{
		//	we insert an hash entry without stagger
		//	return value
		//	1:	insert successful, entry_id is written
		//	0:	this voxel already in hash table, don't need to insert, entry_id is written
		//	-2:	empty entry cannot be found, insert failed
		//	-4:	no more voxel block can be allocated

		int bucket_id = hashing_func(X, Y, Z, hash_table_size);

		//	we must wait until we successfully locked the bucket
		while( ! LockBucket(bucket_id) );

		int bucket_start, bucket_end, hash_list_end;
		entry_id = GetHashEntryId(X, Y, Z, bucket_id, bucket_start, bucket_end, hash_list_end);

		//	if this block is already allocated, return 0, indicating this entry already exist
		if( entry_id >= 0 )	{
			UnlockBucket(bucket_id);
			printf("error: InsertHashEntry, entry already exist, %d, %d, %d\n", X, Y, Z);
			return 0;
		}

		//	if this block is not allocated yet, search an empty space in the bucket
		int insert_entry_id = bucket_start;
		for(; insert_entry_id<=bucket_end; insert_entry_id++){
			if( hash_entry[insert_entry_id].pointer < 0 ){
				int ret = SetHashEntryAndAllocVoxelBlock(insert_entry_id, 0, X, Y, Z);
				entry_id = insert_entry_id;
				UnlockBucket(bucket_id);
				return ret;
			}
		}

		//	the bucket is full, search after the bucket until an empty entry, and insert into the back of the linked list
		insert_entry_id = GetNextNonBucketEndHashEntryId(bucket_end);
		int insert_bucket_id = GetBucketId(insert_entry_id);
		const int count_guard = 1000;	//	the max distance to search
		int count = 0;
		int locked_insert_bucket = 0;
		while( count < count_guard ){
			if( LockBucket(insert_bucket_id) ){// lock the bucket of potential insert
				locked_insert_bucket = 1;
				if( hash_entry[insert_entry_id].pointer >= 0 ){
					locked_insert_bucket = 0;
					UnlockBucket(insert_bucket_id);
				}
				else//	we found an empty entry position, in an locked bucket
					break;
			}

			insert_entry_id = GetNextNonBucketEndHashEntryId(insert_entry_id);
			insert_bucket_id = GetBucketId(insert_entry_id);
			count ++;
		}

		//	we didn't find an empty entry
		if( count == count_guard ){
			if( locked_insert_bucket )
				UnlockBucket(insert_bucket_id);
			UnlockBucket(bucket_id);
			printf("error: InsertHashEntry, cannot find empty entry, %d, %d, %d\n", X, Y, Z);
			return -2;
		}

		//	we have found the entry, lock the bucket and insert now
		if( ! SetHashEntryAndAllocVoxelBlock(insert_entry_id, 0, X, Y, Z) ){
			UnlockBucket(insert_bucket_id);
			UnlockBucket(bucket_id);
			return -4;
		}

		hash_entry[hash_list_end].offset = GetOffset(hash_list_end, insert_entry_id);
		entry_id = insert_entry_id;

		UnlockBucket(insert_bucket_id);
		UnlockBucket(bucket_id);

		return 1;
	}

	__device__ __forceinline__ int DeleteHashEntry(int X, int Y, int Z) const{
		//	this function will not terminate until the target entry is deleted

		//	return value
		//	1:	delete successful
		//	0:	cannot find this hash entry

		int bucket_id = hashing_func(X, Y, Z, hash_table_size);

		//	we must wait until we successfully locked the bucket
		while( ! LockBucket(bucket_id) );

		int bucket_start, bucket_end, hash_list_end;
		int entry_id = GetHashEntryId(X, Y, Z, bucket_id, bucket_start, bucket_end, hash_list_end);

		//	if this block is doesn't exist
		if( entry_id < 0 ){
			UnlockBucket(bucket_id);
			return 0;
		}

		//	find the bucket that this entry lies
		int delete_bucket_id = GetBucketId(entry_id);

		//	if this entry lies in its own bucket
		if( delete_bucket_id == bucket_id ){
			//	if this entry is a head of a linked list, copy the next element and delete it
			if( ( entry_id - 1 == bucket_id * bucket_size ) && ( hash_entry[entry_id].offset != 0 ) ){
				//	we have to lock this bucket and the bucket containing the next entry
				int next_entry_id = GetOffsetHashEntryId( entry_id, hash_entry[entry_id].offset );
				int new_offset = hash_entry[next_entry_id].offset ? hash_entry[entry_id].offset + hash_entry[next_entry_id].offset : 0;
				int ret = ClearHashEntryAndFreeVoxelBlock(entry_id);
				hash_entry[entry_id] = hash_entry[next_entry_id];
				hash_entry[entry_id].offset = new_offset;
				hash_entry[next_entry_id].pointer = -1;
				UnlockBucket(bucket_id);
				return ret;
			}
			//	else delete directly, no lock is needed since it doesn't affect any others
			else{
				int ret = ClearHashEntryAndFreeVoxelBlock(entry_id);
				UnlockBucket(bucket_id);
				return ret;
			}
		}
		//	this entry is one element inside the linked list
		else{
			//	we have successfully locked the bucket, now delete the entry
			int new_offset = hash_entry[entry_id].offset ? hash_entry[hash_list_end].offset + hash_entry[entry_id].offset : 0;
			hash_entry[hash_list_end].offset = new_offset;
			int ret = ClearHashEntryAndFreeVoxelBlock(entry_id);
			UnlockBucket(bucket_id);
			return ret;
		}
	}

};


__global__ void initHashBucketAtomicLock(PtrSz<unsigned int> hash_bucket_atomic_lock);



#endif	//	__VOXEL_BLOCK_HASH_TABLE_H__