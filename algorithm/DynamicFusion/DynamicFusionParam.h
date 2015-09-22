#pragma once

namespace dfusion
{
	struct Param
	{
		Param();

		/** *****************************************************
		* tsdf volume related
		* ******************************************************/
		int volume_resolution[3];
		int voxels_per_meter;

		/** *****************************************************
		* marching cube related
		* ******************************************************/

		// pyramed level: level = 1 means MC will be performed on a x2 coarse lvel
		int marching_cube_level;

		// process the volume in sequential tiles to prevent too large memory
		int marching_cube_tile_size;

		// ratio*tile_size^3 relates the size of temporary buffer size allocated in MC
		// make it larger if related warning reported
		float marching_cube_max_activeVoxel_ratio;

		// by default 0, used when extract triangles
		float marching_cube_isoValue;

		/** *****************************************************
		* warp field related
		* ******************************************************/
		float warp_radius_search_epsilon; // meters
		float warp_radius_search_beta;

		/** *****************************************************
		* dynamic fusion related
		* ******************************************************/
		float fusion_max_weight;
		float fusion_lambda;
		float fusion_psi_data;
		float fusion_psi_reg;
		
	};
}