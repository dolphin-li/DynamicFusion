//	******************************************************
//	Microsoft Kinect
//
//	Yizhong Zhang
//	******************************************************
#include <iostream>
#include <string>
#include <windows.h>
#include <math.h>

#include "MicrosoftKinect.h"
using namespace std;


Microsoft_Kinect::Microsoft_Kinect(){
	frame_id	= 0;
	ref_kinect	= NULL;
#ifdef ENABLE_KINECT_10
	pMapper = NULL;
#endif

	dep_width	= 0;
	dep_height	= 0;
	dep_h_fov	= 0.0f;
	dep_v_fov	= 0.0f;
	depth_map	= NULL;

	img_width	= 0;
	img_height	= 0;
	img_h_fov	= 0.0f;
	img_v_fov	= 0.0f;
	image_map	= NULL;
}

int Microsoft_Kinect::InitKinect(){
#ifdef ENABLE_KINECT_10
		return InitKinect10();
#else
		return InitKinect20();
#endif
}

Microsoft_Kinect::~Microsoft_Kinect(){
	FreeSpace();
}

void Microsoft_Kinect::FreeSpace(){
#ifdef ENABLE_KINECT_10
		FreeSpace10();
#else
		FreeSpace20();
#endif
}

int Microsoft_Kinect::GetDepthColorIntoBuffer(dfusion::depthtype* depth, unsigned char* pBGRA, 
	bool map2depth, bool mirror_input)
{
	GetDepthMap(mirror_input);

	// get depth
	if (depth)
	{
		const int n = dep_height*dep_width;
		for (int i = 0; i < n; i++)
		{
#ifdef ENABLE_KINECT_10
			depth[i] = depth_map[i].depth;
#else
			depth[i] = depth_map[i];
#endif
		}
	}

#ifdef ENABLE_KINECT_20
	// ldp: currently colormap not implmented for kinect 2
	// thus we return here when kinect 2
	return 1;
#endif

	GetColorMap(mirror_input);
	// get color
	if (pBGRA)
	{
		memset(pBGRA, 0, dep_width*dep_height * 4 * sizeof(char));
		if (map2depth)
		{
#ifdef ENABLE_KINECT_10
			// Get the coordinates to convert color to depth space
			coordMapping.resize(dep_width*dep_height);
			HRESULT hr = pMapper->MapDepthFrameToColorFrame(
				NUI_IMAGE_RESOLUTION_640x480,
				dep_width * dep_height,
				depth_map,
				NUI_IMAGE_TYPE_COLOR,
				NUI_IMAGE_RESOLUTION_640x480,
				dep_width * dep_height,   // the color coordinates that get set are the same array size as the depth image
				coordMapping.data());
			ErrorCheck(hr, "GetColorMap: get next frame");

			const int* src = (const int*)image_map;
			int* dst = (int*)pBGRA;
			for (int y = 0; y < dep_height; y++)
			{
				for (int x = 0; x < dep_width; x++)
				{
					unsigned int destIndex = y * dep_width + x;

					// calculate index into depth array
					int colorInDepthX = coordMapping[destIndex].x;
					int colorInDepthY = coordMapping[destIndex].y;

					// make sure the depth pixel maps to a valid point in color space
					if (colorInDepthX >= 0 && colorInDepthX < dep_width
						&& colorInDepthY >= 0 && colorInDepthY < dep_height
						&& depth_map[destIndex].depth != 0
						)
					{
						// Calculate index into color array
						unsigned int sourceColorIndex = colorInDepthX + (colorInDepthY * dep_width);

						// Copy color pixel
						dst[destIndex] = src[sourceColorIndex];
					}
				}
			}// end for y
#else
			throw std::exception("kinect 2 color mapper not implemented!");
#endif
		}// end if map2depth
		else
		{
			memcpy(pBGRA, image_map, img_width*img_height * 4);
		}// end if not map2depth

		for (int i = 0; i < img_width*img_height; i++)
			pBGRA[i * 4 + 3] = 255;
	}// end if pBGRA

	return 1;
}

//	interface functions =======================================================
int Microsoft_Kinect::GetColorMap(bool mirror){
#ifdef ENABLE_KINECT_10
	NUI_IMAGE_FRAME imageFrame;

	HRESULT hr = pNuiSensor->NuiImageStreamGetNextFrame(pColorStreamHandle, 0, &imageFrame);
	if( hr == E_NUI_FRAME_NO_DATA ){	//	new frame not arrived yet, this is not an error
		return 0;
	}
	ErrorCheck(hr, "GetColorMap: get next frame");

	NUI_LOCKED_RECT LockedRect;
	hr = imageFrame.pFrameTexture->LockRect(0, &LockedRect, NULL, 0);
	ErrorCheck(hr, "GetColorMap: lock rect");

	for (int y = 0; y < dep_height; y++)
	{
		const BGRA32Pixel* pSrc = (const BGRA32Pixel*)LockedRect.pBits + dep_width * y;
		BGRA32Pixel* pDst = image_map + dep_width * y;
		if (mirror)
		for (int x = 0; x < dep_width; x++)
		{
			pDst[x] = pSrc[dep_width - 1 - x];
			std::swap(pDst[x].nRed, pDst[x].nBlue);
		}
		else
		for (int x = 0; x < dep_width; x++)
		{
			pDst[x] = pSrc[x];
			std::swap(pDst[x].nRed, pDst[x].nBlue);
		}
	}// end for y

	hr = imageFrame.pFrameTexture->UnlockRect(0);
	ErrorCheck(hr, "GetColorMap: unlock rect");

	hr = pNuiSensor->NuiImageStreamReleaseFrame(pColorStreamHandle, &imageFrame);
	ErrorCheck(hr, "GetColorMap: release frame");

	return 1;
#else
	throw std::exception("kinect 2 color map not implemented!");
#endif
}

int Microsoft_Kinect::GetDepthMap(bool mirror)
{
#ifdef ENABLE_KINECT_10
		return GetDepthMap10(mirror);
#else
		return GetDepthMap20();
#endif
}

int	Microsoft_Kinect::ReadAccelerometer(float gravity[3]){
#ifdef ENABLE_KINECT_10
	Vector4 grav;
	HRESULT hr = pNuiSensor->NuiAccelerometerGetCurrentReading(&grav);
	gravity[0] = grav.x;
	gravity[1] = grav.y;
	gravity[2] = grav.z;
	return 1;
#else
	throw std::exception("kinect 2 gravity not implemented!");
#endif
}

//	v1.0 =======================================================
int Microsoft_Kinect::InitKinect10(){
#ifdef ENABLE_KINECT_10
	//	check the number of connected sensor
	int iSensorCount = 0;
	HRESULT hr = NuiGetSensorCount(&iSensorCount);
	ErrorCheck(hr, "MicrosoftKinect::InitKinect10: unable to count connected");

	//	create kinect by given index
	hr = NuiCreateSensorByIndex(0, &pNuiSensor);
	ErrorCheck(hr, "MicrosoftKinect::InitKinect10: unable to connect to kinect by index");

	// Get the status of the sensor, and if connected, then we can initialize it
	hr = pNuiSensor->NuiStatus();
	if (S_OK != hr){
		pNuiSensor->Release();
		std::cout << "error: MicrosoftKinect::InitKinect10: "
			<< "connection invalid, error code: " << hr << std::endl;
		return 0;
	}

	//	initialize parameters 
	int init_flag = NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH;
	hr = pNuiSensor->NuiInitialize(init_flag); 
	ErrorCheck(hr, "MicrosoftKinect::InitKinect10: initialization failed");

	//	create event and streams
	///<	\todo	this part is under construction, currently we just open color and depth stream
	hNextDepthFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	hr = pNuiSensor->NuiImageStreamOpen(
		NUI_IMAGE_TYPE_DEPTH,
		NUI_IMAGE_RESOLUTION_640x480,
		0,
		2,
		hNextDepthFrameEvent,
		&pDepthStreamHandle);
	ErrorCheck(hr, "MicrosoftKinect::InitKinect10: open depth frame");

	hNextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	hr = pNuiSensor->NuiImageStreamOpen(
		NUI_IMAGE_TYPE_COLOR,
		NUI_IMAGE_RESOLUTION_640x480,
		0,
		2,
		hNextColorFrameEvent,
		&pColorStreamHandle );
	ErrorCheck(hr, "MicrosoftKinect::InitKinect10: open color frame");

	// coord mapper
	hr = pNuiSensor->NuiGetCoordinateMapper(&pMapper);
	ErrorCheck(hr, "MicrosoftKinect::InitKinect10: get coord mapper");

	//	---------------------------------------
	//	init class storage
	frame_id		= 0;

	dep_width		= 640;
	dep_height		= 480;
	dep_h_fov		= NUI_CAMERA_DEPTH_NOMINAL_HORIZONTAL_FOV;
	dep_v_fov		= NUI_CAMERA_DEPTH_NOMINAL_VERTICAL_FOV;
	depth_map = new NUI_DEPTH_IMAGE_PIXEL[ dep_width * dep_height ];

	img_width		= 640;
	img_height		= 480;
	img_h_fov		= NUI_CAMERA_COLOR_NOMINAL_HORIZONTAL_FOV;
	img_v_fov		= NUI_CAMERA_COLOR_NOMINAL_VERTICAL_FOV;
	image_map = new BGRA32Pixel[ img_width * img_height ];

	//	check alloc space
	if( !depth_map || !image_map ){
		cout << "error: MicrosoftKinect::InitKinect10, alloc kinect storage space failed" << endl;
		exit(0);
	}

	return 1;
#else
	std::cout << "error: MicrosoftKinect::ENABLE_KINECT_10 not defined, cannot use kinect 1.0" << std::endl;
	return 0;
#endif
}

int Microsoft_Kinect::FreeSpace10(){
#ifdef ENABLE_KINECT_10
	if (NULL != pNuiSensor) {
		pNuiSensor->NuiShutdown();
		pNuiSensor->Release();
	}

	CloseHandle(hNextDepthFrameEvent);
	CloseHandle(hNextColorFrameEvent);
	return 1;
#else
	std::cout << "error: MicrosoftKinect::ENABLE_KINECT_10 not defined, cannot use kinect 1.0" << std::endl;
	return 0;
#endif
}

int Microsoft_Kinect::GetDepthMap10(bool mirror){
#ifdef ENABLE_KINECT_10
	NUI_IMAGE_FRAME imageFrame;

	HRESULT hr = pNuiSensor->NuiImageStreamGetNextFrame(pDepthStreamHandle, 0, &imageFrame);
	if( hr == E_NUI_FRAME_NO_DATA ){	//	new frame not arrived yet, this is not an error
		return 0;
	}
	ErrorCheck(hr, "error: Microsoft_Kinect::GetDepthMap10: get next frame");

	//	---------------	copied from kinect explorer
	BOOL nearMode;
	INuiFrameTexture* pTexture;
	hr = pNuiSensor->NuiImageFrameGetDepthImagePixelFrameTexture(pDepthStreamHandle, &imageFrame, &nearMode, &pTexture);
	if (FAILED(hr)){
		pNuiSensor->NuiImageStreamReleaseFrame(pDepthStreamHandle, &imageFrame);
		return 0;
	}

	//	---------------	
	NUI_LOCKED_RECT LockedRect;
	hr = pTexture->LockRect(0, &LockedRect, NULL, 0);
	ErrorCheck(hr, "error: Microsoft_Kinect::GetDepthMap10, lock rect");
	
	errno_t err = memcpy_s(
		depth_map,
		dep_width*dep_height * sizeof(NUI_DEPTH_IMAGE_PIXEL),
		LockedRect.pBits,
		pTexture->BufferLen());

	for (int y = 0; y < dep_height; y++)
	{
		const NUI_DEPTH_IMAGE_PIXEL* pSrc = (const NUI_DEPTH_IMAGE_PIXEL*)LockedRect.pBits + dep_width * y;
		NUI_DEPTH_IMAGE_PIXEL* pDst = depth_map + dep_width * y;
		if (mirror)
		for (int x = 0; x < dep_width; x++)
			pDst[x] = pSrc[dep_width - 1 - x];
		else
		for (int x = 0; x < dep_width; x++)
			pDst[x] = pSrc[x];
	}// end for y

	hr = pTexture->UnlockRect(0);
	ErrorCheck(hr, "error: Microsoft_Kinect::GetDepthMap10, unlock rect");

	pTexture->Release();

	hr = pNuiSensor->NuiImageStreamReleaseFrame(pDepthStreamHandle, &imageFrame);
	ErrorCheck(hr, "error: Microsoft_Kinect::GetDepthMap10, release frame");

	return 1;
#else
	std::cout << "error: MicrosoftKinect::ENABLE_KINECT_10 not defined, cannot use kinect 1.0" << std::endl;
	return 0;
#endif
}

//	v2.0 =======================================================
int Microsoft_Kinect::InitKinect20(){
#ifdef ENABLE_KINECT_20
	HRESULT hr;

	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	ErrorCheck(hr, "error: Microsoft_Kinect::InitKinect20, GetDefaultKinectSensor");

	if (m_pKinectSensor){
		// Initialize the Kinect and get the depth reader
		IDepthFrameSource* pDepthFrameSource = NULL;

		hr = m_pKinectSensor->Open();

		if (SUCCEEDED(hr)){
			hr = m_pKinectSensor->get_DepthFrameSource(&pDepthFrameSource);
		}

		if (SUCCEEDED(hr)){
			hr = pDepthFrameSource->OpenReader(&m_pDepthFrameReader);
		}

		SafeRelease(pDepthFrameSource);
	}

	if (!m_pKinectSensor || FAILED(hr)){
		std::cout << "error: Microsoft_Kinect::InitKinect20, No ready Kinect found!" << std::endl;
		return 0;
	}

	//	---------------------------------------
	//	init class storage
	frame_id	= 0;

	dep_width		= 512;
	dep_height		= 424;
	dep_h_fov		= 70.6f;
	dep_v_fov		= 60.0f;
	depth_map = new unsigned short[ dep_width * dep_height ];

	img_width		= 640;
	img_height		= 480;
	img_h_fov		= 70.6f;
	img_v_fov		= 60.0f;
	image_map = new BGRA32Pixel[ img_width * img_height ];

	//	check alloc space
	if( !depth_map || !image_map ){
		cout << "error: MicrosoftKinect::InitKinect20, alloc kinect storage space failed" << endl;
		exit(0);
	}

	return 1;

#else
	std::cout << "error: Microsoft_Kinect::InitKinect20, ENABLE_KINECT_20 not defined, cannot use kinect 2.0" << std::endl;
	return 0;
#endif
}

int Microsoft_Kinect::FreeSpace20(){
#ifdef ENABLE_KINECT_20
	return 1;
#else
	std::cout << "error: Microsoft_Kinect::InitKinect20, ENABLE_KINECT_20 not defined, cannot use kinect 2.0" << std::endl;
	return 0;
#endif
}

int Microsoft_Kinect::GetDepthMap20(){
#ifdef ENABLE_KINECT_20
	if (!m_pDepthFrameReader){
		return 0;
	}

	IDepthFrame* pDepthFrame = NULL;

	HRESULT hr = m_pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);

	if (SUCCEEDED(hr)){
		INT64 nTime = 0;
		IFrameDescription* pFrameDescription = NULL;
		int nWidth = 0;
		int nHeight = 0;
		float h_fov = 0.0f;
		float v_fov = 0.0f;
		USHORT nDepthMinReliableDistance = 0;
		USHORT nDepthMaxReliableDistance = 0;
		UINT nBufferSize = 0;
		UINT16 *pBuffer = NULL;

		hr = pDepthFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr)){
			hr = pDepthFrame->get_FrameDescription(&pFrameDescription);
		}

		if (SUCCEEDED(hr)){
			hr = pFrameDescription->get_Width(&nWidth);
		}

		if (SUCCEEDED(hr)){
			hr = pFrameDescription->get_Height(&nHeight);
		}

		if (SUCCEEDED(hr)){
			hr = pFrameDescription->get_HorizontalFieldOfView(&h_fov);
		}

		if (SUCCEEDED(hr)){
			hr = pFrameDescription->get_VerticalFieldOfView(&v_fov);
		}

		if (SUCCEEDED(hr)){
			hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);
		}

		if (SUCCEEDED(hr)){
			hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxReliableDistance);
		}

		if (SUCCEEDED(hr)){
			hr = pDepthFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);            
		}

		//	now, we have depth map in pBuffer
		if( dep_width != nWidth || dep_height != nHeight ){
			dep_width	= nWidth;
			dep_height	= nHeight;
			dep_h_fov	= h_fov;
			dep_v_fov	= v_fov;
			if( depth_map )
				delete depth_map;
			depth_map = new unsigned short[dep_width*dep_height];
			printf("kinect 2 depth: %d %d\n", dep_width, dep_height);
		}

		memcpy( depth_map, pBuffer, sizeof(unsigned short)*dep_width*dep_height);

		//	flip
		for(int j=0; j<dep_height; j++){
			unsigned short* start_ptr	= depth_map + j*dep_width;
			unsigned short* end_ptr		= depth_map + (j+1)*dep_width;
			std::reverse( start_ptr, end_ptr );
		}

		SafeRelease(pFrameDescription);

		SafeRelease(pDepthFrame);

		return 1;
	}
	else{
		SafeRelease(pDepthFrame);

		return 0;
	}

	return 0;
#else
	std::cout << "error: Microsoft_Kinect::GetDepthMap20, ENABLE_KINECT_20 not defined, cannot use kinect 2.0" << std::endl;
	return 0;
#endif
}

//	=======================================================
void Microsoft_Kinect::ErrorCheck(HRESULT hr, string str){
	if (FAILED(hr) ) { 
		std::cout << "error: MicrosoftKinect, " 
			<< str << ", error code: " << std::hex << hr << std::endl;
	}
}

