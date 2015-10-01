#pragma once

#include "definations.h"
#include "DynamicFusionParam.h"
class Camera;
namespace dfusion
{
	class TsdfVolume;
	class RayCaster;
	class GpuMesh;
	class MarchingCubes;
	class WarpField;
	class DynamicFusionProcessor
	{
	public:
		DynamicFusionProcessor();
		~DynamicFusionProcessor();

		void init(Param param);
		void clear();
		void reset();

		void processFrame(const DepthMap& depth);

		// if not use_ray_casting, then use marching_cube
		void shading(const Camera& userCam, LightSource light, ColorMap& img, bool use_ray_casting);

		const WarpField* getWarpField()const;

		void updateParam(const Param& param);
	protected:
		void estimateWarpField();
		void nonRigidTsdfFusion();
		void surfaceExtractionMC();
		void insertNewDeformNodes();
		void updateRegularizationGraph();
		void updateKNNField();

		Tbx::Transfo rigid_align();
		void fusion();
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
		std::vector<DepthMap> m_depth_curr_pyd;
		std::vector<MapArr> m_vmap_curr_pyd;
		std::vector<MapArr> m_nmap_curr_pyd;
		std::vector<DepthMap> m_depth_prev_pyd;
		std::vector<MapArr> m_vmap_prev_pyd;
		std::vector<MapArr> m_nmap_prev_pyd;
		DeviceArray2D<float> m_rigid_gbuf;
		DeviceArray<float> m_rigid_sumbuf;
		float m_rigid_distThre;
		float m_rigid_angleThre_sin;

		/** *********************
		* for non-rigid align
		* **********************/

		// map of verts in canonical view
		MapArr m_vmap_cano;
		// map of normals in canonical view
		MapArr m_nmap_cano;
	};
}