#pragma once

namespace dfusion
{
	struct Param
	{
		Param();
		int volume_resolution[3];
		int voxels_per_meter;
		int marching_cube_level;
		int marching_cube_tile_size;

		float fusion_max_weight;
	};
}