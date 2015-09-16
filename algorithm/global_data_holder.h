#pragma once

#include "util.h"
#include "ObjMesh.h"
#include "mpu\VolumeData.h"
#include "MicrosoftKinect.h"
#include "kinect_util.h"
#include "TsdfVolume.h"
#include "RayCaster.h"
class GlobalDataHolder
{
public:
	void init();

	static void saveDepth(const std::vector<dfusion::depthtype>& depth_h, std::string filename);
	static void loadDepth(std::vector<dfusion::depthtype>& depth_h, std::string filename);
public:
	Microsoft_Kinect m_kinect;
	std::vector<dfusion::depthtype> m_depth_h;
	dfusion::DepthMap m_depth_d;
	dfusion::TsdfVolume m_volume;
	dfusion::RayCaster m_rayCaster;
	dfusion::LightSource m_lights;

	dfusion::ColorMap m_warpedview_shading;
	bool m_view_normalmap;

private:
	mutable ldp::TimeStamp m_timeStamp;
};

extern GlobalDataHolder g_dataholder;