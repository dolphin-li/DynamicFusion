#pragma once

#include "definations.h"
class ObjMesh;
namespace dfusion
{

	/** **********************************************************************
	* function utils
	* ***********************************************************************/
	void mapsToObj(ObjMesh& mesh, const dfusion::MapArr& vmap, const dfusion::MapArr& nmap);

	// jet_Rgb_d = jet((depth_d-shift)/div)
	void calc_temperature_jet(PtrStepSz<depthtype> depth_d, PtrStepSz<uchar4> jetRgb_d, float shift, float div);

	// shading
	void generateImage(const MapArr& vmap, const MapArr& nmap, ColorMap& dst, const LightSource& light);

	// normal to color
	void generateNormalMap(const MapArr& nmap, ColorMap& dst, Mat33 R);
	void generateNormalMap(const MapArr& nmap, ColorMap& dst);

	// a simple copy function
	void copyColorMapToPbo(PtrStepSz<PixelRGBA> src, PtrStepSz<uchar4> dst);

	// depth map smoothing
	void bilateralFilter(const DepthMap& src, DepthMap& dst);

	// compute vertex/normal from kinect depth
	void createVMap(const Intr& intr, const DepthMap& depth, MapArr& vmap);
	void createNMap(const MapArr& vmap, MapArr& nmap);

	// /2 scale
	void pyrDown(const DepthMap& src, DepthMap& dst);

	void resizeVMap(const MapArr& input, MapArr& output);
	void resizeNMap(const MapArr& input, MapArr& output);

	//
	void rigidTransform(MapArr& vmap, MapArr& nmap, Tbx::Transfo T);

	// estimate rigid transform Ax=b
	void estimateCombined(const Mat33& Rcurr, const float3& tcurr,
		const MapArr& vmap_curr, const MapArr& nmap_curr,
		const Mat33& Rprev, const float3& tprev, const Intr& intr,
		const MapArr& vmap_prev, const MapArr& nmap_prev,
		float distThres, float angleThres,
		DeviceArray2D<float>& gbuf, DeviceArray<float>& mbuf,
		float* matrixA_host, float* vectorB_host);

	//
	void computeErrorMap(const MapArr& vmap_live, const MapArr& nmap_live,
		const MapArr& vmap_warp, const MapArr& nmap_warp, ColorMap& errMap,
		Intr intr, float errMap_range, float distThre, float angleThre_sin);
}