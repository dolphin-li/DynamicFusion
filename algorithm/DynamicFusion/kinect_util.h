#pragma once

#include "definations.h"
namespace dfusion
{

	/** **********************************************************************
	* function utils
	* ***********************************************************************/
	// jet_Rgb_d = jet((depth_d-shift)/div)
	void calc_temperature_jet(PtrStepSz<depthtype> depth_d, PtrStepSz<uchar4> jetRgb_d, float shift, float div);

	// shading
	void generateImage(const MapArr& vmap, const MapArr& nmap, ColorMap& dst, const LightSource& light);

	// normal to color
	void generateNormalMap(const MapArr& nmap, ColorMap& dst, Mat33 R);

	// a simple copy function
	void copyColorMapToPbo(PtrStepSz<PixelRGBA> src, PtrStepSz<uchar4> dst);

	// depth map smoothing
	void bilateralFilter(const DepthMap& src, DepthMap& dst);

	// compute vertex/normal from kinect depth
	void createVMap(const Intr& intr, const DepthMap& depth, MapArr& vmap);
	void createNMap(const MapArr& vmap, MapArr& nmap);

	// /2 scale
	void pyrDown(const DepthMap& src, DepthMap& dst);

	//
	void rigidTransform(MapArr& vmap, MapArr& nmap, Tbx::Transfo T);

	// estimate rigid transform Ax=b
	void estimateCombined(const Mat33& Rcurr, const float3& tcurr,
		const MapArr& vmap_curr, const MapArr& nmap_curr,
		const Mat33& Rprev_inv, const float3& tprev, const Intr& intr,
		const MapArr& vmap_g_prev, const MapArr& nmap_g_prev,
		float distThres, float angleThres,
		DeviceArray2D<double>& gbuf, DeviceArray<double>& mbuf,
		double* matrixA_host, double* vectorB_host);
}