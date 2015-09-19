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
	protected:
		void estimateWarpField();
		void nonRigidTsdfFusion();
		void surfaceExtractionMC();
		void insertNewDeformNodes();
		void updateRegularizationGraph();
		void updateKNNField();
	private:
		Param m_param;
		Camera* m_camera;
		TsdfVolume* m_volume;
		RayCaster* m_rayCaster;
		MarchingCubes* m_marchCube;
		GpuMesh* m_canoMesh;
		GpuMesh* m_warpedMesh;
		std::vector<WarpField*> m_framesWarpFields;

		int m_frame_id;
	};
}