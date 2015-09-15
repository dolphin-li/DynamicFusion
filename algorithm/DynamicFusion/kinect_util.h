#pragma once
#include "cuda_utils.h"
#include "device_array.h"
#include "device_functions.h"
namespace dfusion
{
	/** **********************************************************************
	* types
	* ***********************************************************************/
	struct PixelRGB{
		unsigned char r, g, b;
	};

	typedef unsigned short ushort;
	typedef ushort depthtype;
	typedef DeviceArray2D<float> MapArr;
	typedef DeviceArray2D<depthtype> DepthMap;
	typedef DeviceArray2D<PixelRGB> ColorMap;

	enum{
		KINECT_WIDTH = 640,
		KINECT_HEIGHT = 480
	};

	/** **********************************************************************
	* function utils
	* ***********************************************************************/
	// jet_Rgb_d = jet((depth_d-shift)/div)
	void calc_temperature_jet(PtrStepSz<depthtype> depth_d, PtrStepSz<uchar4> jetRgb_d, float shift, float div);
}