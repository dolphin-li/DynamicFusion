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
#include "ObjMesh.h"
#include "VolumeData.h"
#include "CpuGaussNewton.h"
#include "GpuGaussNewtonSolver.h"
namespace dfusion
{
#define DFUSION_SAFE_DELETE(buffer)\
	if (buffer){ delete buffer; buffer = nullptr; }

	Tbx::Mat3 convert(Eigen::Matrix3f A)
	{
		return Tbx::Mat3(A(0, 0), A(0, 1), A(0, 2),
			A(1, 0), A(1, 1), A(1, 2),
			A(2, 0), A(2, 1), A(2, 2));
	}
	Tbx::Vec3 convert(Eigen::Vector3f A)
	{
		return Tbx::Vec3(A[0], A[1], A[2]);
	}
	ldp::Mat4f convert(Tbx::Transfo T)
	{
		ldp::Mat4f A;
		for (int y = 0; y < 4; y++)
		for (int x = 0; x < 4; x++)
			A(y, x) = T[y*4+x];
		return A;
	}

	DynamicFusionProcessor::DynamicFusionProcessor()
	{
		m_camera = nullptr;
		m_volume = nullptr;
		m_rayCaster = nullptr;
		m_marchCube = nullptr;
		m_canoMesh = nullptr;
		m_warpedMesh = nullptr;
		m_warpField = nullptr;
		m_frame_id = 0;
		m_gsSolver = nullptr;
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

		// volume
		m_volume = new TsdfVolume();
		m_volume->init(make_int3(
			m_param.volume_resolution[0], 
			m_param.volume_resolution[1], 
			m_param.volume_resolution[2]),
			1.f/m_param.voxels_per_meter, 
			make_float3(-m_param.volume_resolution[0]*0.5f/m_param.voxels_per_meter, 
			-m_param.volume_resolution[1] * 0.5f / m_param.voxels_per_meter, 
			-KINECT_NEAREST_METER - float(m_param.volume_resolution[2]) / m_param.voxels_per_meter)
			);

		// mesh
		m_canoMesh = new GpuMesh();
		m_warpedMesh = new GpuMesh();

		// marching cube
		m_marchCube = new MarchingCubes();
		m_marchCube->init(m_volume, m_param);

		// ray casting
		m_rayCaster = new RayCaster();
		m_rayCaster->init(*m_volume);

		// warp field
		m_warpField = new WarpField();
		m_warpField->init(m_volume, m_param);

		// GaussNewton solver
		m_gsSolver = new GpuGaussNewtonSolver();

		// maps
		m_depth_curr_pyd.resize(RIGID_ALIGN_PYD_LEVELS);
		m_vmap_curr_pyd.resize(RIGID_ALIGN_PYD_LEVELS);
		m_nmap_curr_pyd.resize(RIGID_ALIGN_PYD_LEVELS);
		m_depth_prev_pyd.resize(RIGID_ALIGN_PYD_LEVELS);
		m_vmap_prev_pyd.resize(RIGID_ALIGN_PYD_LEVELS);
		m_nmap_prev_pyd.resize(RIGID_ALIGN_PYD_LEVELS);
		m_vmap_cano.create(KINECT_HEIGHT, KINECT_WIDTH);
		m_nmap_cano.create(KINECT_HEIGHT, KINECT_WIDTH);
		m_vmap_warp.create(KINECT_HEIGHT, KINECT_WIDTH);
		m_nmap_warp.create(KINECT_HEIGHT, KINECT_WIDTH);

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
		DFUSION_SAFE_DELETE(m_warpedMesh);
		DFUSION_SAFE_DELETE(m_warpField);
		DFUSION_SAFE_DELETE(m_gsSolver);
		m_frame_id = 0;
		m_depth_curr_pyd.clear();
		m_vmap_curr_pyd.clear();
		m_nmap_curr_pyd.clear();
		m_depth_prev_pyd.clear();
		m_vmap_prev_pyd.clear();
		m_nmap_prev_pyd.clear();
		m_vmap_cano.release();
		m_nmap_cano.release();
		m_vmap_warp.release();
		m_nmap_warp.release();
	}

	void DynamicFusionProcessor::updateParam(const Param& param)
	{
		bool reCreate = false;
		bool reSet = false;

		for (int k = 0; k < 3; k++)
		{
			if (m_param.volume_resolution[k] != param.volume_resolution[k])
				reCreate = true;
		}
		if (m_param.voxels_per_meter != param.voxels_per_meter)
			reCreate = true;

		m_param = param;
		if (m_warpField)
			m_warpField->setActiveVisualizeNodeId(m_param.view_activeNode_id);

		if (reCreate)
			init(param);
		else if (reSet)
			reset();
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
		m_warpField->init(m_volume, m_param);

		// solver
		m_gsSolver->reset();

		m_frame_id = 0;
	}

	void DynamicFusionProcessor::processFrame(const DepthMap& depth)
	{
		depth.copyTo(m_depth_input);
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
		if (use_ray_casting)
		{
			Camera cam = *m_camera;
			cam.setModelViewMatrix(userCam.getModelViewMatrix()*m_camera->getModelViewMatrix());
			m_rayCaster->setCamera(cam);
			m_rayCaster->shading(light, img);
		}
		else
		{
			// read passed image pos and return selections/related-knns for visualization
			static int lastX = -1, lastY = -1;
			static WarpField::KnnIdx lastKnn;
			static float3 lastCano;
			const int x = m_param.view_click_vert_xy[0];
			const int y = m_param.view_click_vert_xy[1];
			WarpField::KnnIdx* knnPtr = nullptr;
			WarpField::KnnIdx knnIdx = make_ushort4(WarpField::MaxNodeNum, WarpField::MaxNodeNum,
				WarpField::MaxNodeNum, WarpField::MaxNodeNum);
			float3* canoPosPtr = nullptr;
			float3 canoPos = make_float3(0,0,0);
			if (lastX != x || lastY != y)
			{
				lastX = x;
				lastY = y;
				if (x >= 0 && x < m_vmap_cano.cols()
					&& y >= 0 && y < m_vmap_cano.rows())
				{
					cudaSafeCall(cudaMemcpy(&canoPos, ((char*)m_vmap_cano.ptr())
						+ m_vmap_cano.step()*y + x*sizeof(float4)
						, sizeof(float3), cudaMemcpyDeviceToHost), "copy cano pos");
					knnIdx = m_warpField->getKnnAt(canoPos);
					knnPtr = &knnIdx;
					canoPosPtr = &canoPos;
					printf("knnIdx: %d %d %d %d\n", knnIdx.x, knnIdx.y, knnIdx.z, knnIdx.w);
				}
				lastKnn = knnIdx;
				lastCano = canoPos;
			}
			else
			{
				knnPtr = &lastKnn;
				canoPosPtr = &lastCano;
			}

			// render
			m_warpedMesh->renderToImg(userCam, light, img, m_param, m_warpField,
				&m_vmap_curr_pyd[0], &m_vmap_warp, &m_nmap_curr_pyd[0],
				&m_nmap_warp, m_canoMesh, canoPosPtr, knnPtr, &m_kinect_intr);
			//img.create(m_nmap_cano.rows(), m_nmap_cano.cols());
			//generateNormalMap(m_vmap_cano, img);
		}
	}

	void DynamicFusionProcessor::shadingCanonical(const Camera& userCam, LightSource light, 
		ColorMap& img, bool use_ray_casting)
	{
		if (use_ray_casting)
		{
			Camera cam = *m_camera;
			cam.setModelViewMatrix(userCam.getModelViewMatrix());
			m_rayCaster->setCamera(cam);
			m_rayCaster->shading(light, img);
		}
		else
		{
			m_canoMesh->renderToImg(userCam, light, img, m_param, nullptr, 
				nullptr, nullptr, nullptr, nullptr);
		}
	}

	void DynamicFusionProcessor::shadingCurrentErrorMap(ColorMap& img, float errorMapRange)
	{
		computeErrorMap(m_vmap_curr_pyd[0], m_nmap_curr_pyd[0], m_vmap_warp, m_nmap_warp,
			img, m_kinect_intr, errorMapRange, m_param.fusion_nonRigid_distThre,
			m_param.fusion_nonRigid_angleThreSin);
	}

	const WarpField* DynamicFusionProcessor::getWarpField()const
	{
		return m_warpField;
	}

	void DynamicFusionProcessor::estimateWarpField()
	{
		Tbx::Transfo rigid = rigid_align();

		m_warpField->set_rigidTransform(rigid);

		if (m_frame_id == 0)
			return;

		// 0. create visibility map of the current warp view
		m_warpedMesh->renderToCanonicalMaps(*m_camera, m_canoMesh, m_vmap_cano, m_nmap_cano);
		m_warpField->warp(m_vmap_cano, m_nmap_cano, m_vmap_warp, m_nmap_warp);

		if (!m_param.fusion_enable_nonRigidSolver)
			return;


		// icp iteration
		m_gsSolver->init(m_warpField, m_vmap_cano, m_nmap_cano, m_param, m_kinect_intr);
		for (int icp_iter = 0; icp_iter < m_param.fusion_nonRigidICP_maxIter; icp_iter++)
		{
			// 1. find correspondence
			// m_warpedMesh->renderToCanonicalMaps(*m_camera, m_canoMesh, m_vmap_cano, m_nmap_cano);

			// 2. Gauss-Newton Optimization
#if 0
			CpuGaussNewton solver;
			solver.init(m_warpField, m_vmap_cano, m_nmap_cano, m_param, m_kinect_intr);
			Eigen::VectorXf debugX;
			debugX.resize(m_warpField->getNumAllNodes()*6);
			debugX.setRandom();
			debugX *= 0.1f;
			m_gsSolver->debug_set_init_x(debugX.data(), debugX.size());
			solver.debug_set_init_x(debugX.data(), debugX.size());

			solver.findCorr(m_vmap_curr_pyd[0], m_nmap_curr_pyd[0], m_vmap_warp, m_nmap_warp);
			solver.solve(m_param.fusion_post_rigid_factor);

			ldp::tic();
			m_gsSolver->solve(m_vmap_curr_pyd[0], m_nmap_curr_pyd[0], m_vmap_warp, m_nmap_warp);
			cudaThreadSynchronize();
			ldp::toc("gpu solver");
			m_gsSolver->debug_print();
			system("pause");
#else
			m_gsSolver->solve(m_vmap_curr_pyd[0], m_nmap_curr_pyd[0], m_vmap_warp, m_nmap_warp);
#endif

			// 3. update warped mesh and render for visiblity
			m_warpField->warp(m_vmap_cano, m_nmap_cano, m_vmap_warp, m_nmap_warp);
		}// end for icp_iter

		// finally, re-factor out the rigid part across all nodes

	}

	void DynamicFusionProcessor::nonRigidTsdfFusion()
	{
		fusion();
	}

	void DynamicFusionProcessor::surfaceExtractionMC()
	{
		m_marchCube->run(*m_canoMesh);
		m_warpField->warp(*m_canoMesh, *m_warpedMesh);
		m_warpedMesh->renderToDepth(*m_camera, m_depth_prev_pyd[0]);
		createVMap(m_kinect_intr, m_depth_prev_pyd[0], m_vmap_prev_pyd[0]);
		createNMap(m_vmap_prev_pyd[0], m_nmap_prev_pyd[0]);

		for (int i = 1; i < RIGID_ALIGN_PYD_LEVELS; ++i){
			resizeVMap(m_vmap_prev_pyd[i - 1], m_vmap_prev_pyd[i]);
			resizeNMap(m_nmap_prev_pyd[i - 1], m_nmap_prev_pyd[i]);
		}
	}

	void DynamicFusionProcessor::insertNewDeformNodes()
	{
		m_warpField->updateWarpNodes(*m_canoMesh);
	}

	void DynamicFusionProcessor::updateRegularizationGraph()
	{
		// done in insertNewDeformNodes()
	}

	void DynamicFusionProcessor::updateKNNField()
	{
		// done in insertNewDeformNodes()
	}

	Tbx::Transfo DynamicFusionProcessor::rigid_align()
	{
		//bilateralFilter(m_depth_input, m_depth_curr_pyd[0]);
		m_depth_input.copyTo(m_depth_curr_pyd[0]);
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
			return Tbx::Transfo().identity();

		// now estimate rigid transform
		Tbx::Transfo c2v = m_warpField->get_rigidTransform().fast_invert();
		Tbx::Mat3	c2v_Rprev = c2v.get_mat3();
		Tbx::Vec3	c2v_tprev = c2v.get_translation();

		Tbx::Mat3	c2v_Rcurr = c2v_Rprev;
		Tbx::Vec3	c2v_tcurr = c2v_tprev;

		const int icp_iterations[] = { 2, 4, 10 };
		for (int level_index = RIGID_ALIGN_PYD_LEVELS - 1; level_index >= 0; --level_index)
		{
			MapArr& vmap_curr = m_vmap_curr_pyd[level_index];
			MapArr& nmap_curr = m_nmap_curr_pyd[level_index];
			MapArr& vmap_prev = m_vmap_prev_pyd[level_index];
			MapArr& nmap_prev = m_nmap_prev_pyd[level_index];

			int iter_num = icp_iterations[level_index];
			for (int iter = 0; iter < iter_num; ++iter)
			{
				Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A;
				Eigen::Matrix<float, 6, 1> b;

				estimateCombined(convert(c2v_Rcurr), convert(c2v_tcurr), vmap_curr, nmap_curr,
					convert(c2v_Rprev), convert(c2v_tprev), m_kinect_intr(level_index),
					vmap_prev, nmap_prev, m_param.fusion_rigid_distThre, m_param.fusion_rigid_angleThreSin, 
					m_rigid_gbuf, m_rigid_sumbuf, A.data(), b.data());

				//checking nullspace
				float det = A.determinant();
				if (fabs(det) < std::numeric_limits<float>::epsilon() || _isnan(det))
				{
					if (_isnan(det))
						std::cout << "qnan" << std::endl;
					else
						std::cout << det << std::endl;
					return c2v.fast_invert();
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
				c2v_tcurr = convert(Rinc) * c2v_tcurr + convert(tinc);
				c2v_Rcurr = convert(Rinc) * c2v_Rcurr;
			}
		}

		return Tbx::Transfo(c2v_Rcurr, c2v_tcurr).fast_invert();
	}
}