#pragma once

namespace dfusion
{
	struct Param
	{
		Param();

		void save(const char* filename)const;
		void load(const char* filename);

		/** *****************************************************
		* tsdf volume related
		* ******************************************************/
		int volume_resolution[3];
		int voxels_per_meter;
		void set_voxels_per_meter(int v);

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

		// in tsdf volume, marching cube is only performed when weight is large enough
		float marchingCube_min_valied_weight;

		/** *****************************************************
		* warp field related
		* ******************************************************/
		int warp_knn_k_eachlevel[4];
		// meters, for radius search
		// the larger, the less freedom for warping
		float warp_radius_search_epsilon; 
		void set_warp_radius_search_epsilon(float v);

		// param \dw in the paper for each node
		// the larger, the less smoothness and more rigid
		float warp_param_dw;
		// we fusion, we expect more smooth output
		float warp_param_dw_for_fusion;

		// scale = \dw_{level+1} / \dw_{level}
		// the larger, the more rigid
		float warp_param_dw_lvup_scale;

		// controls the softness of lv0-lv1 graph:
		// exp(-softness * dist)
		float warp_param_softness;

		// scalar of epsilon for different levels, as in the paper \beta
		// the larger, the less graph level-nodes.
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
		// the larger, the more time invariant
		float fusion_max_weight;

		// the larger, the more rigid
		float fusion_lambda;

		// Tukey penalty coeffs for the data term
		float fusion_psi_data;

		// Huber penalty coeffs for the regularization term
		float fusion_psi_reg;
		
		// 
		int fusion_nonRigidICP_maxIter;
		int fusion_rigid_ICP_iter[3];
		float fusion_rigid_distThre;
		float fusion_rigid_angleThreSin;
		float fusion_nonRigid_distThre;
		float fusion_nonRigid_angleThreSin;

		//
		int fusion_GaussNewton_maxIter;

		// if this is not zero, then we use this 
		// fixed step size instead of line search
		// this may cause divergenced system, but 
		// more efficient on the GPU.
		float fusion_GaussNewton_fixedStep;

		// a small diagonal regularization term, 
		// used to prevent numerical issues, which may cause #nan.
		float fusion_GaussNewton_diag_regTerm;

		// if true, the after each non-linear optimization,
		// the common rigid transformation will be factored out 
		// and merged into the rigid part.
		bool fusion_post_rigid_factor;

		// debugging related
		bool fusion_dumping_each_frame;
		bool fusion_enable_nonRigidSolver;
		bool fusion_enable_rigidSolver;
		bool fusion_loading_mode;
		int fusion_dumping_max_frame;
		bool mirror_input;
		int load_frameIndx_plus_num;
		bool solver_enable_nan_check;
		bool graph_single_level;
		float graph_remove_small_components_ratio;

		/** *****************************************************
		* visualization related
		* ******************************************************/
		bool view_no_rigid;
		bool view_show_mesh;
		bool view_show_nodes;
		bool view_show_graph;
		bool view_show_corr;
		int view_show_graph_level;
		float view_errorMap_range; // in meters
		int view_activeNode_id;
		int view_click_vert_xy[2];
		bool view_show_color;

		bool view_autoreset;
		int view_autoreset_seconds;
	};
}