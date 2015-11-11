#pragma once

#include "definations.h"
#include "DynamicFusionParam.h"
class Camera;
namespace dfusion
{
//#define SPARSE_VOLUME_TESTING
	class TsdfVolume;
	class RayCaster;
	class GpuMesh;
	class MarchingCubes;
	class WarpField;
	class GpuGaussNewtonSolver;
	class SparseVolume;
	class DynamicFusionProcessor
	{
	public:
		DynamicFusionProcessor();
		~DynamicFusionProcessor();

		void init(Param param);
		void clear();
		void reset();

		void save(const char* volume_name);
		void load(const char* volume_name);

		void processFrame(const DepthMap& depth, const ColorMap& color);

		// if not use_ray_casting, then use marching_cube
		void shading(const Camera& userCam, LightSource light, ColorMap& img, bool use_ray_casting);
		void shadingCanonical(const Camera& userCam, LightSource light, ColorMap& img, bool use_ray_casting);
		void shadingCurrentErrorMap(ColorMap& img, float errorMapRange);

		const WarpField* getWarpField()const;
		WarpField* getWarpField();
		const TsdfVolume* getVolume()const;
		TsdfVolume* getVolume();
		const GpuGaussNewtonSolver* getSolver()const;
		GpuGaussNewtonSolver* getSolver();
		GpuMesh* getWarpedMesh(){ return m_warpedMesh; }

		void updateParam(const Param& param);

		int getFrameId()const{ return m_frame_id; }

		bool hasRawDepth()const{ return m_depth_input.rows() > 0; }
		const MapArr& getRawDepthNormal()const{ return m_nmap_curr_pyd.at(0); }
	protected:
		void estimateWarpField();
		void nonRigidTsdfFusion();
		void surfaceExtractionMC();
		void insertNewDeformNodes();
		void updateRegularizationGraph();
		void updateKNNField();

		void eroseColor(const ColorMap& src, ColorMap& dst, int nRadius);

		Tbx::Transfo rigid_align();
		void fusion();
	protected:
		//===================Sparse Volume Testing===========================
#ifdef SPARSE_VOLUME_TESTING
		void VoxelBlockAllocation(const DepthMap& depth_float_frame_d);
		void VisibleVoxelBlockSelection();
		void VoxelBlockUpdate(const DepthMap& depth_float_frame_d);
		void OutActiveRegionVoxelBlockSelection();
		int GPU2HostStreaming();
		int Host2GPUStreaming();
#endif
	private:
		Param m_param;
		Camera* m_camera;
		TsdfVolume* m_volume;
		RayCaster* m_rayCaster;
		MarchingCubes* m_marchCube;
		GpuMesh* m_canoMesh;
		GpuMesh* m_warpedMesh;
		WarpField* m_warpField;
		Intr m_kinect_intr;

		int m_frame_id;

		/** *********************
		* for rigid align
		* **********************/
		enum{
			RIGID_ALIGN_PYD_LEVELS = 3
		};
		DepthMap m_depth_input;
		ColorMap m_color_input;
		ColorMap m_color_tmp;
		std::vector<DepthMap> m_depth_curr_pyd;
		std::vector<MapArr> m_vmap_curr_pyd;
		std::vector<MapArr> m_nmap_curr_pyd;
		std::vector<DepthMap> m_depth_prev_pyd;
		std::vector<MapArr> m_vmap_prev_pyd;
		std::vector<MapArr> m_nmap_prev_pyd;
		DeviceArray2D<float> m_rigid_gbuf;
		DeviceArray<float> m_rigid_sumbuf;

		/** *********************
		* for non-rigid align
		* **********************/

		// map of verts in canonical/warped view
		DeviceArray2D<float4> m_vmap_cano;
		DeviceArray2D<float4> m_vmap_warp;
		// map of normals in canonical/warped view
		DeviceArray2D<float4> m_nmap_cano;
		DeviceArray2D<float4> m_nmap_warp;

		GpuGaussNewtonSolver* m_gsSolver;

		//===============Sparse Volume Testing====================
#ifdef SPARSE_VOLUME_TESTING
		SparseVolume* m_sparseVolume;
#endif
	};
}