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

		// meters, for radius search
		float warp_radius_search_epsilon; 

		// param \dw in the paper for each node
		float warp_param_dw;

		// scalar of epsilon for different levels, as in the paper \beta
		float warp_radius_search_beta;

		// there may be not necessary to use the full resolution points.
		// thus we may firstly subsample the points and then generate nodes
		// this may be much faster
		int warp_point_step_before_update_node; 

		// to avoid noise, only node-grid contains enough points will be considered
		float warp_valid_point_num_each_node;

		/** *****************************************************
		* dynamic fusion related
		* ******************************************************/
		float fusion_max_weight;
		float fusion_lambda;
		float fusion_psi_data;
		float fusion_psi_reg;
		int fusion_nonRigidICP_maxIter;
		float fusion_rigid_distThre;
		float fusion_rigid_angleThreSin;
		float fusion_nonRigid_distThre;
		float fusion_nonRigid_angleThreSin;

		/** *****************************************************
		* visualization related
		* ******************************************************/
		bool view_show_mesh;
		bool view_show_nodes;
		bool view_show_graph;
		bool view_show_corr;
		int view_show_graph_level;
	};
}