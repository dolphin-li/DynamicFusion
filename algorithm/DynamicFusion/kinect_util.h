#pragma once

#include "defininations.h"
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

	//
	void copyColorMapToPbo(PtrStepSz<PixelRGBA> src, PtrStepSz<uchar4> dst);
}