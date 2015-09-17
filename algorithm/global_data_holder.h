#pragma once

#include "util.h"
#include "ObjMesh.h"
#include "mpu\VolumeData.h"
#include "MicrosoftKinect.h"
#include "kinect_util.h"
#include "TsdfVolume.h"
#include "RayCaster.h"
#include "MarchingCubes.h"
#include "GpuMesh.h"
class GlobalDataHolder
{
public:
	void init();

	static void saveDepth(const std::vector<dfusion::depthtype>& depth_h, std::string filename);
	static void loadDepth(std::vector<dfusion::depthtype>& depth_h, std::string filename);

	static void meshCopy(const dfusion::GpuMesh& gmesh, ObjMesh& mesh);
public:
	Microsoft_Kinect m_kinect;
	std::vector<dfusion::depthtype> m_depth_h;
	dfusion::DepthMap m_depth_d;
	dfusion::RayCaster m_rayCaster;
	dfusion::MarchingCubes m_marchCube;
	dfusion::LightSource m_lights;

	dfusion::ColorMap m_warpedview_shading;
	bool m_view_normalmap;

	dfusion::TsdfVolume m_volume;
	dfusion::GpuMesh m_mesh;
private:
	mutable ldp::TimeStamp m_timeStamp;
};

extern GlobalDataHolder g_dataholder;