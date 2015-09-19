#include "DynamicFusionProcessor.h"
#include "GpuMesh.h"
#include "RayCaster.h"
#include "MarchingCubes.h"
#include "TsdfVolume.h"
#include "Camera.h"
#include "WarpField.h"
#include "fmath.h"

namespace dfusion
{
#define DFUSION_SAFE_DELETE(buffer)\
	if (buffer){ delete buffer; buffer = nullptr; }

	DynamicFusionProcessor::DynamicFusionProcessor()
	{
		m_camera = nullptr;
		m_volume = nullptr;
		m_rayCaster = nullptr;
		m_marchCube = nullptr;
		m_canoMesh = nullptr;
		m_frame_id = 0;
	}

	DynamicFusionProcessor::~DynamicFusionProcessor()
	{
		clear();
	}

	void DynamicFusionProcessor::init(Param param)
	{
		clear();

		m_param = param;
		// allocation----------------------

		// camera
		m_camera = new Camera();
		m_camera->setViewPort(0, KINECT_WIDTH, 0, KINECT_HEIGHT);
		m_camera->setPerspective(KINECT_DEPTH_V_FOV, float(KINECT_WIDTH) / KINECT_HEIGHT,
			KINECT_NEAREST_METER, 30.f);
		const float l = m_camera->getViewPortLeft();
		const float r = m_camera->getViewPortRight();
		const float t = m_camera->getViewPortTop();
		const float b = m_camera->getViewPortBottom();
		const float	f = (b - t) * 0.5f / tanf(m_camera->getFov() * fmath::DEG_TO_RAD * 0.5f);
		m_kinect_intr.fx = f;
		m_kinect_intr.fy = f;
		m_kinect_intr.cx = (l + r) / 2;
		m_kinect_intr.cy = (t + b) / 2;

		// volume
		m_volume = new TsdfVolume();
		m_volume->init(make_int3(m_param.volume_resolution[0], 
			m_param.volume_resolution[1], m_param.volume_resolution[2]),
			1.f/m_param.voxels_per_meter, make_float3(0.f, 0.f, KINECT_NEAREST_METER));

		// mesh
		m_canoMesh = new GpuMesh();
		m_warpedMesh = new GpuMesh();

		// marching cube
		m_marchCube = new MarchingCubes();
		m_marchCube->init(m_volume, m_param.marching_cube_tile_size, m_param.marching_cube_level);

		// ray casting
		m_rayCaster = new RayCaster();
		m_rayCaster->init(*m_volume);

		// maps
		m_depth_curr_pyd.resize(RIGID_ALIGN_PYD_LEVELS);
		m_vmap_curr_pyd.resize(RIGID_ALIGN_PYD_LEVELS);
		m_nmap_curr_pyd.resize(RIGID_ALIGN_PYD_LEVELS);
		m_depth_prev_pyd.resize(RIGID_ALIGN_PYD_LEVELS);
		m_vmap_prev_pyd.resize(RIGID_ALIGN_PYD_LEVELS);
		m_nmap_prev_pyd.resize(RIGID_ALIGN_PYD_LEVELS);

		// finally reset----------------------
		reset();
	}

	void DynamicFusionProcessor::clear()
	{
		DFUSION_SAFE_DELETE(m_camera);
		DFUSION_SAFE_DELETE(m_volume);
		DFUSION_SAFE_DELETE(m_rayCaster);
		DFUSION_SAFE_DELETE(m_marchCube);
		DFUSION_SAFE_DELETE(m_canoMesh);
		for (size_t i = 0; i < m_framesWarpFields.size(); i++)
			delete m_framesWarpFields[i];
		m_framesWarpFields.clear();
		m_frame_id = 0;
		m_depth_curr_pyd.clear();
		m_vmap_curr_pyd.clear();
		m_nmap_curr_pyd.clear();
		m_depth_prev_pyd.clear();
		m_vmap_prev_pyd.clear();
		m_nmap_prev_pyd.clear();
	}

	void DynamicFusionProcessor::reset()
	{
		// camera
		m_camera->setModelViewMatrix(ldp::Mat4f().eye());

		// volume
		m_volume->reset();

		// mesh
		m_canoMesh->release();
		m_warpedMesh->release();

		// warp fields
		for (size_t i = 0; i < m_framesWarpFields.size(); i++)
			delete m_framesWarpFields[i];
		m_framesWarpFields.clear();
		m_framesWarpFields.reserve(30000);

		m_frame_id = 0;
	}

	void DynamicFusionProcessor::processFrame(const DepthMap& depth)
	{
		m_depth_input = depth;
		estimateWarpField();
		nonRigidTsdfFusion();
		surfaceExtractionMC();
		insertNewDeformNodes();
		updateRegularizationGraph();
		updateKNNField();
	}

	void DynamicFusionProcessor::shading(const Camera& userCam, LightSource light, 
		ColorMap& img, bool use_ray_casting)
	{
		//debug
		generateImage(m_vmap_curr_pyd[2], m_nmap_curr_pyd[2], img, light);
		return;

		Camera cam = *m_camera;
		cam.setModelViewMatrix(userCam.getModelViewMatrix()*m_camera->getModelViewMatrix());

		if (use_ray_casting)
		{
			m_rayCaster->setCamera(cam);
			m_rayCaster->shading(light, img);
		}
		else
		{
			m_warpedMesh->renderToImg(cam, light, img);
		}
	}

	void DynamicFusionProcessor::estimateWarpField()
	{
		rigid_align();
	}

	void DynamicFusionProcessor::nonRigidTsdfFusion()
	{

	}

	void DynamicFusionProcessor::surfaceExtractionMC()
	{
		m_marchCube->run(*m_canoMesh);
		m_warpedMesh->copyFrom(*m_canoMesh);
	}

	void DynamicFusionProcessor::insertNewDeformNodes()
	{

	}

	void DynamicFusionProcessor::updateRegularizationGraph()
	{

	}

	void DynamicFusionProcessor::updateKNNField()
	{

	}

	void DynamicFusionProcessor::rigid_align()
	{
		bilateralFilter(m_depth_input, m_depth_curr_pyd[0]);
		createVMap(m_kinect_intr(0), m_depth_curr_pyd[0], m_vmap_curr_pyd[0]);
		createNMap(m_vmap_curr_pyd[0], m_nmap_curr_pyd[0]);

		//	create pyramid
		for (int i = 1; i < RIGID_ALIGN_PYD_LEVELS; ++i)
			pyrDown(m_depth_curr_pyd[i - 1], m_depth_curr_pyd[i]);

		//	calculate point cloud and normal map
		for (int i = 0; i < RIGID_ALIGN_PYD_LEVELS; ++i)
		{
			//	opengl camera coordinate, -z is camera direction
			createVMap(m_kinect_intr(i), m_depth_curr_pyd[i], m_vmap_curr_pyd[i]);	
			createNMap(m_vmap_curr_pyd[i], m_nmap_curr_pyd[i]);
		}

		//	if it is the first frame, no volume to align, so stop here
		if (m_frame_id == 0)
			return;
	}
}