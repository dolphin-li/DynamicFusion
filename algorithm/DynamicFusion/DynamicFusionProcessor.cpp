#include "DynamicFusionProcessor.h"
#include "GpuMesh.h"
#include "RayCaster.h"
#include "MarchingCubes.h"
#include "TsdfVolume.h"
#include "Camera.h"
#include "WarpField.h"
#include "fmath.h"
#include <eigen\Dense>
#include <eigen\Geometry>

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
		m_kinect_intr = Intr(f, f, (l + r) / 2, (t + b) / 2);
		m_rigid_distThre = 0.10f;							//	meter
		m_rigid_angleThre_sin = sin(20.f*3.14159254f / 180.f);	//	sin of angle

		// volume
		m_volume = new TsdfVolume();
		m_volume->init(make_int3(
			m_param.volume_resolution[0], 
			m_param.volume_resolution[1], 
			m_param.volume_resolution[2]),
			1.f/m_param.voxels_per_meter, 
			make_float3(-m_param.volume_resolution[0]*0.5f/m_param.voxels_per_meter, 
			-m_param.volume_resolution[1] * 0.5f / m_param.voxels_per_meter, 
			-KINECT_NEAREST_METER)
			);

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
		return;
		m_depth_input = depth;
		estimateWarpField();
		nonRigidTsdfFusion();
		surfaceExtractionMC();
		insertNewDeformNodes();
		updateRegularizationGraph();
		updateKNNField();
		m_frame_id++;
	}

	void DynamicFusionProcessor::shading(const Camera& userCam, LightSource light, 
		ColorMap& img, bool use_ray_casting)
	{
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
		Tbx::Transfo rigid = rigid_align();

		m_framesWarpFields.push_back(new WarpField());
		m_framesWarpFields.back()->set_rigidTransform(rigid);
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

	Tbx::Mat3 convert(Eigen::Matrix3f A)
	{
		return Tbx::Mat3(A(0,0), A(0,1), A(0,2),
						A(1,0), A(1,1), A(1,2),
						A(2,0), A(2,1), A(2,2));
	}
	Tbx::Vec3 convert(Eigen::Vector3f A)
	{
		return Tbx::Vec3(A[0], A[1], A[2]);
	}

	Tbx::Transfo DynamicFusionProcessor::rigid_align()
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

#if 0
		// debug
		m_depth_curr_pyd[0].copyTo(m_depth_prev_pyd[0]);
		for (int i = 1; i < RIGID_ALIGN_PYD_LEVELS; ++i)
			pyrDown(m_depth_prev_pyd[i - 1], m_depth_prev_pyd[i]);

		Tbx::Transfo test_T = Tbx::Transfo::identity();
		Tbx::Quat_cu test_q = Tbx::Quat_cu(Tbx::Vec3(1,0,0), 3.8*3.1415926f/180.f);
		test_T.set_mat3(test_q.to_matrix3());
		test_T.set_translation(Tbx::Vec3(0.003f, 0.005f, 0.002f));

		//printf("test_input---------------------------------------------------:\n");
		//test_T.print();

		//	calculate point cloud and normal map
		for (int i = 0; i < RIGID_ALIGN_PYD_LEVELS; ++i)
		{
			m_vmap_curr_pyd[i].copyTo(m_vmap_prev_pyd[i]);
			m_nmap_curr_pyd[i].copyTo(m_nmap_prev_pyd[i]);
			rigidTransform(m_vmap_curr_pyd[i], m_nmap_curr_pyd[i], test_T.fast_invert());
		}
		if (m_frame_id)
			m_framesWarpFields.back()->set_rigidTransform(Tbx::Transfo::identity());
#endif

		//	if it is the first frame, no volume to align, so stop here
		if (m_frame_id == 0)
			return Tbx::Transfo().identity();

		// now estimate rigid transform
		Tbx::Transfo w2c = m_framesWarpFields.back()->get_rigidTransform();
		Tbx::Transfo c2w = w2c.fast_invert();
		Tbx::Mat3	Rprev = c2w.get_mat3();
		Tbx::Vec3	tprev = c2w.get_translation();
		Tbx::Mat3	Rprev_inv = Rprev.inverse();			//Rprev.t();

		Tbx::Mat3	Rcurr = Rprev;
		Tbx::Vec3	tcurr = tprev;

		const int icp_iterations[] = { 4, 5, 10 };
		for (int level_index = RIGID_ALIGN_PYD_LEVELS - 1; level_index >= 0; --level_index)
		{
			MapArr& vmap_curr = m_vmap_curr_pyd[level_index];
			MapArr& nmap_curr = m_nmap_curr_pyd[level_index];
			MapArr& vmap_g_prev = m_vmap_prev_pyd[level_index];
			MapArr& nmap_g_prev = m_nmap_prev_pyd[level_index];

			int iter_num = icp_iterations[level_index];
			for (int iter = 0; iter < iter_num; ++iter)
			{
				Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
				Eigen::Matrix<double, 6, 1> b;

				estimateCombined(convert(Rcurr), convert(tcurr), vmap_curr, nmap_curr, 
					convert(Rprev_inv), convert(tprev), m_kinect_intr(level_index),
					vmap_g_prev, nmap_g_prev, m_rigid_distThre, m_rigid_angleThre_sin, 
					m_rigid_gbuf, m_rigid_sumbuf, A.data(), b.data());

				//checking nullspace
				double det = A.determinant();
				if (fabs(det) < 1e-15 || _isnan(det))
				{
					if (_isnan(det))
						std::cout << "qnan" << std::endl;
					else
						std::cout << det << std::endl;
					return w2c;
				}

				Eigen::Matrix<float, 6, 1> result = A.llt().solve(b).cast<float>();

				float alpha = result(0);
				float beta = result(1);
				float gamma = result(2);

				Eigen::Matrix3f Rinc = (Eigen::Matrix3f)Eigen::AngleAxisf(gamma, Eigen::Vector3f::UnitZ()) *
					Eigen::AngleAxisf(beta, Eigen::Vector3f::UnitY()) *
					Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX());
				Eigen::Vector3f tinc = result.tail<3>();

				//compose
				tcurr = convert(Rinc) * tcurr + convert(tinc);
				Rcurr = convert(Rinc) * Rcurr;

#if 0
				// debug
				Tbx::Quat_cu q(Rcurr);
				Tbx::Vec3 axis;
				float angle;
				q.to_angleAxis(axis, angle);
				printf("abc: %d %d %f %f %f\n", level_index, iter, alpha*fmath::RAD_TO_DEG, 
					beta*fmath::RAD_TO_DEG, gamma*fmath::RAD_TO_DEG);
				printf("combined: %f %f %f %f\n", angle*fmath::RAD_TO_DEG, axis.x, axis.y, axis.z);
#endif
			}
		}

		return Tbx::Transfo(Rcurr, tcurr);
	}
}