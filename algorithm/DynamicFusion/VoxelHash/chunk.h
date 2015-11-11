/***********************************************************/
/**	\file
	\brief		data structure for chunk
	\details	
	\author		Yizhong Zhang
	\date		12/12/2013
*/
/***********************************************************/
#ifndef __CHUNK_H__
#define __CHUNK_H__

#include <list>
#include "voxel_hashing_internal.h"


class Chunk{
public:
	std::vector<HashEntry>	hash_entry;
	std::list<VoxelBlock>	voxel_block;

public:
	inline void AddData(const HashEntry& entry, const VoxelBlock& block){
		hash_entry.push_back(entry);
		voxel_block.push_back(block);
	}
};




#endif	//	__CHUNK_H__