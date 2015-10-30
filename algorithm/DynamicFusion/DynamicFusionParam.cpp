#include "DynamicFusionParam.h"
#include <math.h>
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
		voxels_per_meter = 387;

		/** *****************************************************
		* marching cube related
		* ******************************************************/
		marching_cube_level = 0;
		marching_cube_tile_size = 256;
		marching_cube_max_activeVoxel_ratio = 0.2;
		marching_cube_isoValue = 0.f;
		marchingCube_min_valied_weight = 2.f;

		/** *****************************************************
		* warp field related
		* ******************************************************/
		set_warp_radius_search_epsilon(0.025);
		warp_param_dw = warp_radius_search_epsilon * 1.733 * 0.5f; // sqrt(3)/2
		warp_param_dw_for_fusion = warp_param_dw * 0.5f; // sqrt(3)/2
		warp_radius_search_beta = 2;
		warp_param_dw_lvup_scale = 0.01f;
		warp_point_step_before_update_node = 1;

		/** *****************************************************
		* dynamic fusion related
		* ******************************************************/
		fusion_max_weight = 512;
		fusion_lambda = 3000;
		fusion_psi_data = 0.01;
		fusion_psi_reg = 0.0001;
		fusion_rigid_distThre = 0.1f; // meter
		fusion_rigid_ICP_iter[0] = 8; // coarse level
		fusion_rigid_ICP_iter[1] = 4;
		fusion_rigid_ICP_iter[2] = 2; // finest level
		fusion_rigid_angleThreSin = sin(45.f*3.14159254f / 180.f);
		fusion_nonRigid_distThre = 0.03f; // meter
		fusion_nonRigid_angleThreSin = sin(90.f*3.14159254f / 180.f);

		fusion_nonRigidICP_maxIter = 2;
		fusion_GaussNewton_maxIter = 2;
		fusion_GaussNewton_diag_regTerm = 1e-3;
		fusion_GaussNewton_fixedStep = 0.0;// 5;

		// debuging related
		fusion_dumping_each_frame = false;
		fusion_loading_mode = true;
		fusion_enable_nonRigidSolver = true;
		fusion_enable_rigidSolver = true;
		fusion_post_rigid_factor = true;
		fusion_dumping_max_frame = 800;
		mirror_input = true; 
		load_frameIndx_plus_num = 1;

		/** *****************************************************
		* visualization related
		* ******************************************************/
		view_no_rigid = false;
		view_show_mesh = true;
		view_show_nodes = false;
		view_show_graph = false;
		view_show_corr = false;
		view_show_graph_level = 0;
		view_errorMap_range = 0.01;
		view_activeNode_id = -1;
		view_click_vert_xy[0] = view_click_vert_xy[1] = -1;

		view_autoreset = false;
		view_autoreset_seconds = 20;
	}

	void Param::set_voxels_per_meter(int v)
	{
		voxels_per_meter = v;
		set_warp_radius_search_epsilon(warp_radius_search_epsilon);
	}

	void Param::set_warp_radius_search_epsilon(float v)
	{
		warp_radius_search_epsilon = v;
		warp_valid_point_num_each_node = 300 * pow(v / 0.025f * voxels_per_meter / 387.f, 3);
	}
}