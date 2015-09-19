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
#include "DynamicFusionParam.h"
#include "DynamicFusionProcessor.h"
class GlobalDataHolder
{
public:
	void init();

	static void saveDepth(const std::vector<dfusion::depthtype>& depth_h, std::string filename);
	static void loadDepth(std::vector<dfusion::depthtype>& depth_h, std::string filename);
public:
	dfusion::DynamicFusionProcessor m_processor;
	Microsoft_Kinect m_kinect;
	dfusion::Param m_dparam;
	dfusion::LightSource m_lights;

	std::vector<dfusion::depthtype> m_depth_h;
	dfusion::DepthMap m_depth_d;
	dfusion::ColorMap m_warpedview_shading;

	// the following is used for debugging/visualizing loaded volumes.
	dfusion::RayCaster m_rayCaster;
	dfusion::MarchingCubes m_marchCube;
	dfusion::TsdfVolume m_volume;
	dfusion::GpuMesh m_mesh;
private:
	mutable ldp::TimeStamp m_timeStamp;
};

extern GlobalDataHolder g_dataholder;