/***********************************************************/
/**	\file
	\brief		Microsoft Kinect with Microsoft Kinect SDK
	\author		Yizhong Zhang
	\date		11/15/2012
*/
/***********************************************************/
#ifndef __MICROSOFT_KINECT_H__
#define __MICROSOFT_KINECT_H__

//	the following control version of kinect
#define ENABLE_KINECT_10
//#define ENABLE_KINECT_20

#include <iostream>
#include <string>
#include <windows.h>
#include <math.h>
#include <vector>

#ifdef ENABLE_KINECT_10
#	include <NuiApi.h>
#	pragma comment(lib, "Kinect10.lib")
#endif

#ifdef ENABLE_KINECT_20
#	include <Kinect.h>
#	pragma comment(lib, "kinect20.lib")
#endif

struct Point3D{
	float X;
	float Y;
	float Z;
};

struct RGB24Pixel{
	unsigned char nRed;
	unsigned char nGreen;
	unsigned char nBlue;
};

struct BGRA32Pixel{
	unsigned char nBlue;
	unsigned char nGreen;
	unsigned char nRed;
	unsigned char alpha;
};


class Microsoft_Kinect{
public:
	//	constructor and destructor
	Microsoft_Kinect();
	int InitKinect(int version = 1);
	~Microsoft_Kinect();
	void FreeSpace();

	// the size of buffer must be 640*480 and continues.
	// if depth/pBGRA==nullptr, then ignored.
	int GetDepthColorIntoBuffer(unsigned short* depth, unsigned char* pBGRA, bool map2depth = true);

	//	get data
	int GetDepthMap();
	int GetColorMap();
	int CalculatePointCloud(float* xyz_ptr);
	int ReadAccelerometer(float gravity[3]);

	int InitKinect(Microsoft_Kinect* another_kinect = NULL, int version = 1);
	int CopyDepth20to10();

public:
	//	current frame id
	int kinect_version_id;
	int frame_id;
	Microsoft_Kinect* ref_kinect;

	//	depth data
	int dep_width, dep_height;
	float dep_h_fov, dep_v_fov;
	NUI_DEPTH_IMAGE_PIXEL*	depth_map;

 	//	color data
	int img_width, img_height;
	float img_h_fov, img_v_fov;
	BGRA32Pixel*		image_map;

	INuiCoordinateMapper* pMapper;
	std::vector<NUI_COLOR_IMAGE_POINT> coordMapping;
#ifdef ENABLE_KINECT_10
	//	Kinect SDK Interface, v1
	INuiSensor* pNuiSensor;
	HANDLE		pColorStreamHandle;
	HANDLE		pDepthStreamHandle;
	HANDLE		hNextColorFrameEvent;
	HANDLE		hNextDepthFrameEvent;
#endif

#ifdef ENABLE_KINECT_20
	//	Kinect interface v2.0
	IKinectSensor*          m_pKinectSensor;		// Current Kinect
    IDepthFrameReader*      m_pDepthFrameReader;	// Depth reader
#endif

private:
	//	kinect v1.0 functions
	int InitKinect10();
	int FreeSpace10();
	int GetDepthMap10();

	//	kinect v2.0 functions
	int InitKinect20();
	int FreeSpace20();
	int GetDepthMap20();

	void ErrorCheck( HRESULT status, std::string str );
};


template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
    if (pInterfaceToRelease != NULL)
    {
        pInterfaceToRelease->Release();
        pInterfaceToRelease = NULL;
    }
}

#endif
