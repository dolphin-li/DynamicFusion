#include "DynamicFusionParam.h"

namespace dfusion
{
	Param::Param()
	{
		/** *****************************************************
		* tsdf volume related
		* ******************************************************/
		volume_resolution[0] = 256;
		volume_resolution[1] = 256;
		volume_resolution[2] = 256;
		voxels_per_meter = 256;

		/** *****************************************************
		* marching cube related
		* ******************************************************/
		marching_cube_level = 0;
		marching_cube_tile_size = 256;
		marching_cube_max_activeVoxel_ratio = 0.2;
		marching_cube_isoValue = 0.f;

		/** *****************************************************
		* warp field related
		* ******************************************************/
		warp_radius_search_epsilon = 0.025;
		warp_radius_search_beta = 4;

		/** *****************************************************
		* dynamic fusion related
		* ******************************************************/
		fusion_max_weight = 128;
		fusion_lambda = 200;
		fusion_psi_data = 0.01;
		fusion_psi_reg = 0.0001;
	}
}