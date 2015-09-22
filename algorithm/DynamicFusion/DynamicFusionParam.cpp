#include "DynamicFusionParam.h"

namespace dfusion
{
	Param::Param()
	{
		volume_resolution[0] = 256;
		volume_resolution[1] = 256;
		volume_resolution[2] = 256;
		voxels_per_meter = 256;

		marching_cube_level = 0;
		marching_cube_tile_size = 256;
		marching_cube_max_activeVoxel_ratio = 0.2;
		marching_cube_isoValue = 0.f;

		fusion_max_weight = 128;
	}
}