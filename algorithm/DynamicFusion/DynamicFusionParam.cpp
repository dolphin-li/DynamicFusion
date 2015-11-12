#include "DynamicFusionParam.h"
#include <math.h>
#include <stdio.h>
#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "definations.h"
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
		voxels_per_meter = 512;

		/** *****************************************************
		* marching cube related
		* ******************************************************/
		marching_cube_level = 0;
		marching_cube_tile_size = 256;
		marching_cube_max_activeVoxel_ratio = 0.2;
		marching_cube_isoValue = 0.f;
		marchingCube_min_valied_weight = 1.f;

		/** *****************************************************
		* warp field related
		* ******************************************************/
		// cannot be larger than warpField::knnK
		warp_knn_k_eachlevel[0] = 4;	// graph-pixel association
		warp_knn_k_eachlevel[1] = 4; // finest graph
		warp_knn_k_eachlevel[2] = KnnK;
		warp_knn_k_eachlevel[3] = KnnK;
		set_warp_radius_search_epsilon(0.025);
		warp_param_softness = 0.5;
		warp_radius_search_beta = 2;
		warp_param_dw_lvup_scale = 0.01f;
		warp_point_step_before_update_node = 1;

		/** *****************************************************
		* dynamic fusion related
		* ******************************************************/
		fusion_max_weight = 512;
		fusion_lambda = 200;
		fusion_psi_data = 0.01;
		fusion_psi_reg = 0.0001;
		fusion_rigid_distThre = 0.1f; // meter
		fusion_rigid_ICP_iter[0] = 8; // coarse level
		fusion_rigid_ICP_iter[1] = 4;
		fusion_rigid_ICP_iter[2] = 0; // finest level
		fusion_rigid_angleThreSin = sin(45.f*3.14159254f / 180.f);
		fusion_nonRigid_distThre = 0.03f; // meter
		fusion_nonRigid_angleThreSin = sin(90.f*3.14159254f / 180.f);

		fusion_nonRigidICP_maxIter = 3;
		fusion_GaussNewton_maxIter = 2;
		fusion_GaussNewton_diag_regTerm = 1e-5;
		fusion_GaussNewton_fixedStep = 0.;

		// debuging related
		fusion_dumping_each_frame = false;
		fusion_loading_mode = false;
		fusion_enable_nonRigidSolver = true;
		fusion_enable_rigidSolver = true;
		fusion_post_rigid_factor = true;
		fusion_dumping_max_frame = 8000;
		mirror_input = false; 
		load_frameIndx_plus_num = 1;
		solver_enable_nan_check = false;
		graph_single_level = false;
		graph_remove_small_components_ratio = 10.1f; //code not ready, use this param>=1.f to disable

		if (graph_single_level)
		{
			fusion_lambda = 300;
			fusion_GaussNewton_fixedStep = 0.5f;
			warp_param_softness = 0;
			fusion_GaussNewton_diag_regTerm = 1e-6;
		}

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
		view_show_color = false;

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
		warp_valid_point_num_each_node = 300.f * pow(v / 0.025f * voxels_per_meter / 387.f, 3);
		warp_param_dw = warp_radius_search_epsilon * 1.733 * 0.5f; // sqrt(3)/2
		warp_param_dw_for_fusion = warp_param_dw * 0.5f; // sqrt(3)/2
	}

#define VAR2STR(a) #a


#define WRITE_ONE(a)\
	stm << VAR2STR(a:) << a << std::endl;
#define WRITE_2(a)\
	stm << VAR2STR(a:) << a[0] << " " << a[1] << std::endl;
#define WRITE_3(a)\
	stm << VAR2STR(a:) << a[0] << " " << a[1] << " " << a[2] << std::endl;
#define WRITE_4(a)\
	stm << VAR2STR(a:) << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << std::endl;

	void Param::save(const char* filename)const
	{
		std::ofstream stm(filename);
		if (stm.fail())
			throw std::exception("IO error: save param failed!");

		/** *****************************************************
		* tsdf volume related
		* ******************************************************/
		WRITE_3(volume_resolution);
		WRITE_ONE(voxels_per_meter);

		/** *****************************************************
		* marching cube related
		* ******************************************************/
		WRITE_ONE(marching_cube_level);
		WRITE_ONE(marching_cube_tile_size);
		WRITE_ONE(marching_cube_max_activeVoxel_ratio);
		WRITE_ONE(marching_cube_isoValue);
		WRITE_ONE(marchingCube_min_valied_weight);

		/** *****************************************************
		* warp field related
		* ******************************************************/
		// cannot be larger than warpField::knnK
		WRITE_4(warp_knn_k_eachlevel);
		WRITE_ONE(warp_radius_search_epsilon);
		WRITE_ONE(warp_valid_point_num_each_node);
		WRITE_ONE(warp_param_dw);
		WRITE_ONE(warp_param_dw_for_fusion);
		WRITE_ONE(warp_radius_search_beta);
		WRITE_ONE(warp_param_dw_lvup_scale);
		WRITE_ONE(warp_param_softness);
		WRITE_ONE(warp_point_step_before_update_node);

		/** *****************************************************
		* dynamic fusion related
		* ******************************************************/
		WRITE_ONE(fusion_max_weight);
		WRITE_ONE(fusion_lambda);
		WRITE_ONE(fusion_psi_data);
		WRITE_ONE(fusion_psi_reg);
		WRITE_ONE(fusion_rigid_distThre);
		WRITE_ONE(fusion_rigid_distThre);
		WRITE_3(fusion_rigid_ICP_iter);
		WRITE_ONE(fusion_rigid_angleThreSin);
		WRITE_ONE(fusion_nonRigid_distThre);
		WRITE_ONE(fusion_nonRigid_angleThreSin);
		WRITE_ONE(fusion_nonRigidICP_maxIter);
		WRITE_ONE(fusion_GaussNewton_maxIter);
		WRITE_ONE(fusion_GaussNewton_diag_regTerm);
		WRITE_ONE(fusion_GaussNewton_fixedStep);

		// debuging related
		WRITE_ONE(fusion_dumping_each_frame);
		WRITE_ONE(fusion_loading_mode);
		WRITE_ONE(fusion_enable_nonRigidSolver);
		WRITE_ONE(fusion_enable_rigidSolver);
		WRITE_ONE(fusion_post_rigid_factor);
		WRITE_ONE(fusion_dumping_max_frame);
		WRITE_ONE(mirror_input);
		WRITE_ONE(load_frameIndx_plus_num);
		WRITE_ONE(solver_enable_nan_check);
		WRITE_ONE(graph_single_level);
		WRITE_ONE(graph_remove_small_components_ratio);

		/** *****************************************************
		* visualization related
		* ******************************************************/
		WRITE_ONE(view_no_rigid);
		WRITE_ONE(view_show_mesh);
		WRITE_ONE(view_show_nodes);
		WRITE_ONE(view_show_graph);
		WRITE_ONE(view_show_corr);
		WRITE_ONE(view_show_graph_level);
		WRITE_ONE(view_errorMap_range);
		WRITE_ONE(view_activeNode_id);
		WRITE_2(view_click_vert_xy);
		WRITE_ONE(view_show_color);
		WRITE_ONE(view_autoreset);
		WRITE_ONE(view_autoreset_seconds);

		stm.close();
	}

	static std::string getLineLabel(std::string& buffer)
	{
		std::string s;
		int pos = buffer.find_first_of(':');
		if (pos < buffer.size())
		{
			s = buffer.substr(0, pos + 1);
			buffer = buffer.substr(pos + 1);
		}
		return s;
	}

#define READ_ONE_IF(a)\
	if (lineLabel == VAR2STR(a:))\
	strstm >> a;
#define READ_3_IF(a)\
	if (lineLabel == VAR2STR(a:))\
	strstm >> a[0] >> a[1] >> a[2];
#define READ_ONE_ELSE_IF(a)\
	else if (lineLabel == VAR2STR(a:))\
	strstm >> a;
#define READ_2_ELSE_IF(a)\
	else if (lineLabel == VAR2STR(a:))\
	strstm >> a[0] >> a[1];
#define READ_3_ELSE_IF(a)\
	else if (lineLabel == VAR2STR(a:))\
	strstm >> a[0] >> a[1] >> a[2];
#define READ_4_ELSE_IF(a)\
	else if (lineLabel == VAR2STR(a:))\
	strstm >> a[0] >> a[1] >> a[2] >> a[3];

	void Param::load(const char* filename)
	{
		std::ifstream stm(filename);
		if (stm.fail())
			throw std::exception("IO error: Param::load() failed!");

		std::string lineBuffer;
		while (!stm.eof())
		{
			std::getline(stm, lineBuffer);
			std::string lineLabel = getLineLabel(lineBuffer);
			std::istringstream strstm(lineBuffer);

			/** *****************************************************
			* tsdf volume related
			* ******************************************************/
			READ_3_IF(volume_resolution)
				READ_ONE_ELSE_IF(voxels_per_meter)

				/** *****************************************************
				* marching cube related
				* ******************************************************/
				READ_ONE_ELSE_IF(marching_cube_level)
				READ_ONE_ELSE_IF(marching_cube_tile_size)
				READ_ONE_ELSE_IF(marching_cube_max_activeVoxel_ratio)
				READ_ONE_ELSE_IF(marching_cube_isoValue)
				READ_ONE_ELSE_IF(marchingCube_min_valied_weight)

				/** *****************************************************
				* warp field related
				* ******************************************************/
				READ_4_ELSE_IF(warp_knn_k_eachlevel)
				READ_ONE_ELSE_IF(warp_radius_search_epsilon)
				READ_ONE_ELSE_IF(warp_valid_point_num_each_node)
				READ_ONE_ELSE_IF(warp_param_dw)
				READ_ONE_ELSE_IF(warp_param_dw_for_fusion)
				READ_ONE_ELSE_IF(warp_radius_search_beta)
				READ_ONE_ELSE_IF(warp_param_dw_lvup_scale)
				READ_ONE_ELSE_IF(warp_param_softness)
				READ_ONE_ELSE_IF(warp_point_step_before_update_node)

				/** *****************************************************
				* dynamic fusion related
				* ******************************************************/
				READ_ONE_ELSE_IF(fusion_max_weight)
				READ_ONE_ELSE_IF(fusion_lambda)
				READ_ONE_ELSE_IF(fusion_psi_data)
				READ_ONE_ELSE_IF(fusion_psi_reg)
				READ_ONE_ELSE_IF(fusion_rigid_distThre)
				READ_ONE_ELSE_IF(fusion_rigid_distThre)
				READ_3_ELSE_IF(fusion_rigid_ICP_iter)
				READ_ONE_ELSE_IF(fusion_rigid_angleThreSin)
				READ_ONE_ELSE_IF(fusion_nonRigid_distThre)
				READ_ONE_ELSE_IF(fusion_nonRigid_angleThreSin)
				READ_ONE_ELSE_IF(fusion_nonRigidICP_maxIter)
				READ_ONE_ELSE_IF(fusion_GaussNewton_maxIter)
				READ_ONE_ELSE_IF(fusion_GaussNewton_diag_regTerm)
				READ_ONE_ELSE_IF(fusion_GaussNewton_fixedStep)

				// debuging related
				READ_ONE_ELSE_IF(fusion_dumping_each_frame)
				READ_ONE_ELSE_IF(fusion_loading_mode)
				READ_ONE_ELSE_IF(fusion_enable_nonRigidSolver)
				READ_ONE_ELSE_IF(fusion_enable_rigidSolver)
				READ_ONE_ELSE_IF(fusion_post_rigid_factor)
				READ_ONE_ELSE_IF(fusion_dumping_max_frame)
				READ_ONE_ELSE_IF(mirror_input)
				READ_ONE_ELSE_IF(load_frameIndx_plus_num)
				READ_ONE_ELSE_IF(solver_enable_nan_check)
				READ_ONE_ELSE_IF(graph_single_level)
				READ_ONE_ELSE_IF(graph_remove_small_components_ratio)

				/** *****************************************************
				* visualization related
				* ******************************************************/
				READ_ONE_ELSE_IF(view_no_rigid)
				READ_ONE_ELSE_IF(view_show_mesh)
				READ_ONE_ELSE_IF(view_show_nodes)
				READ_ONE_ELSE_IF(view_show_graph)
				READ_ONE_ELSE_IF(view_show_corr)
				READ_ONE_ELSE_IF(view_show_graph_level)
				READ_ONE_ELSE_IF(view_errorMap_range)
				READ_ONE_ELSE_IF(view_activeNode_id)
				READ_2_ELSE_IF(view_click_vert_xy)
				READ_ONE_ELSE_IF(view_show_color)
				READ_ONE_ELSE_IF(view_autoreset)
				READ_ONE_ELSE_IF(view_autoreset_seconds)
		}
		set_voxels_per_meter(voxels_per_meter);
		set_warp_radius_search_epsilon(warp_radius_search_epsilon);

		stm.close();

		printf("param loaded successfully!\n");
	}
}