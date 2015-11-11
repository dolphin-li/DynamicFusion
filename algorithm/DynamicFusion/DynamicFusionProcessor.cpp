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
#include "SparseVolume.h"
namespace dfusion
{
//#define ENABLE_CPU_DEBUG
//#define ENABLE_BILATERAL_PRE_FILTER

	static void check(const char* msg)
	{
		cudaSafeCall(cudaThreadSynchronize());
		printf("%s\n", msg);
	}

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
#ifdef SPARSE_VOLUME_TESTING
		m_sparseVolume = nullptr;
#endif
	}

	DynamicFusionProcessor::~DynamicFusionProcessor()
	{
		clear();
		DFUSION_SAFE_DELETE(m_camera);
		DFUSION_SAFE_DELETE(m_volume);
		DFUSION_SAFE_DELETE(m_rayCaster);
		DFUSION_SAFE_DELETE(m_marchCube);
		DFUSION_SAFE_DELETE(m_canoMesh);
		DFUSION_SAFE_DELETE(m_warpedMesh);
		DFUSION_SAFE_DELETE(m_warpField);
		DFUSION_SAFE_DELETE(m_gsSolver);
#ifdef SPARSE_VOLUME_TESTING
		DFUSION_SAFE_DELETE(m_sparseVolume);
#endif
	}

	void DynamicFusionProcessor::init(Param param)
	{
		clear();

		m_param = param;
		// allocation----------------------

		// camera
		if (m_camera == nullptr)
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
		if (m_volume == nullptr)
			m_volume = new TsdfVolume();
		m_volume->init(make_int3(
			m_param.volume_resolution[0], 
			m_param.volume_resolution[1], 
			m_param.volume_resolution[2]),
			1.f/m_param.voxels_per_meter, 
			make_float3(-(m_param.volume_resolution[0]-1)*0.5f/m_param.voxels_per_meter, 
			-(m_param.volume_resolution[1]-1) * 0.5f / m_param.voxels_per_meter, 
			-KINECT_NEAREST_METER - float(m_param.volume_resolution[2]-1) / m_param.voxels_per_meter)
			);
#ifdef SPARSE_VOLUME_TESTING
		if (m_sparseVolume == nullptr)
			m_sparseVolume = new SparseVolume();
		m_sparseVolume->init(make_int3(
			m_param.volume_resolution[0],
			m_param.volume_resolution[1],
			m_param.volume_resolution[2]),
			1.f / m_param.voxels_per_meter,
			make_float3(-(m_param.volume_resolution[0] - 1)*0.5f / m_param.voxels_per_meter,
			-(m_param.volume_resolution[1] - 1) * 0.5f / m_param.voxels_per_meter,
			-KINECT_NEAREST_METER - float(m_param.volume_resolution[2] - 1) / m_param.voxels_per_meter)
			);
#endif

		// mesh
		if (m_canoMesh == nullptr)
			m_canoMesh = new GpuMesh();
		if (m_warpedMesh == nullptr)
			m_warpedMesh = new GpuMesh();
		m_canoMesh->setShowColor(m_param.view_show_color);
		m_warpedMesh->setShowColor(m_param.view_show_color);

		// marching cube
		if (m_marchCube == nullptr)
			m_marchCube = new MarchingCubes();
		m_marchCube->init(m_volume, m_param);

		// ray casting
		if (m_rayCaster == nullptr)
			m_rayCaster = new RayCaster();
		m_rayCaster->init(*m_volume);

		// warp field
		if (m_warpField == nullptr)
			m_warpField = new WarpField();
		m_warpField->init(m_volume, m_param);

		// GaussNewton solver
		if (m_gsSolver == nullptr)
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
		if (m_camera)
			m_camera->reset();
		if (m_volume)
			m_volume->reset();
		if (m_rayCaster)
			m_rayCaster->clear();
		if (m_canoMesh)
			m_canoMesh->release();
		if (m_warpedMesh)
			m_warpedMesh->release();
		if (m_warpField)
			m_warpField->clear();
		if (m_gsSolver)
			m_gsSolver->reset();
		m_frame_id = 0;
		m_depth_input.release();
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
		m_rigid_sumbuf.release();
		m_rigid_gbuf.release();
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

		if (m_canoMesh)
			m_canoMesh->setShowColor(m_param.view_show_color);
		if (m_warpedMesh)
			m_warpedMesh->setShowColor(m_param.view_show_color);

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

	void DynamicFusionProcessor::processFrame(const DepthMap& depth,
		const ColorMap& color)
	{
		depth.copyTo(m_depth_input);

		color.copyTo(m_color_input);
		//eroseColor(color, m_color_input, 1);

		Tbx::Transfo rigid = rigid_align();
		m_warpField->set_rigidTransform(rigid);

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
			static KnnIdx lastKnn;
			static float3 lastCano;
			const int x = m_param.view_click_vert_xy[0];
			const int y = m_param.view_click_vert_xy[1];
			KnnIdx* knnPtr = nullptr;
			KnnIdx knnIdx = make_knn(WarpField::MaxNodeNum);
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

					printf("knnIdx[%d]: ", KnnK);
					for (int k = 0; k < KnnK; k++)
						printf("%d\n", knn_k(knnIdx, k));
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
				&m_nmap_warp, m_canoMesh, canoPosPtr, knnPtr, &m_kinect_intr, true,
				m_param.view_no_rigid);
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
			m_canoMesh->renderToImg(userCam, light, img, m_param, m_warpField,
				&m_vmap_curr_pyd[0], &m_vmap_warp, &m_nmap_curr_pyd[0],
				&m_nmap_warp, m_canoMesh, nullptr, nullptr, &m_kinect_intr, false, false);
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

	WarpField* DynamicFusionProcessor::getWarpField()
	{
		return m_warpField;
	}

	const TsdfVolume* DynamicFusionProcessor::getVolume()const
	{
		return m_volume;
	}

	TsdfVolume* DynamicFusionProcessor::getVolume()
	{
		return m_volume;
	}

	const GpuGaussNewtonSolver* DynamicFusionProcessor::getSolver()const
	{
		return m_gsSolver;
	}

	GpuGaussNewtonSolver* DynamicFusionProcessor::getSolver()
	{
		return m_gsSolver;
	}

	void DynamicFusionProcessor::save(const char* volume_name)
	{
		m_volume->save(volume_name);

		std::string path, purename, ext;
		ldp::fileparts(volume_name, path, purename, ext);
		std::string name1 = fullfile(path, purename + ".warpfield");
		m_warpField->save(name1.c_str());
		std::string name2 = fullfile(path, purename + ".param.txt");
		m_param.save(name2.c_str());
	}

	void DynamicFusionProcessor::load(const char* volume_name)
	{
		int3 old_res = m_volume->getResolution();
		m_volume->load(volume_name);
		int3 new_res = m_volume->getResolution();
		if (old_res.x != new_res.x || old_res.y != new_res.y || old_res.z != new_res.z)
			throw std::exception("error in loading volumes, size not matched with pre-allocated!");
		
		std::string path, purename, ext;
		ldp::fileparts(volume_name, path, purename, ext);
		std::string name1 = fullfile(path, purename + ".warpfield");
		m_warpField->load(name1.c_str());
		
		surfaceExtractionMC(); 
		m_warpedMesh->renderToCanonicalMaps(*m_camera, m_canoMesh, m_vmap_cano, m_nmap_cano);
		m_warpField->warp(m_vmap_cano, m_nmap_cano, m_vmap_warp, m_nmap_warp);
		m_frame_id = 1; // !=0 to prevent trivial case.
	}

	void DynamicFusionProcessor::estimateWarpField()
	{
		if (m_frame_id == 0)
			return;

		// 0. create visibility map of the current warp view
		m_warpedMesh->renderToCanonicalMaps(*m_camera, m_canoMesh, m_vmap_cano, m_nmap_cano);
		m_warpField->warp(m_vmap_cano, m_nmap_cano, m_vmap_warp, m_nmap_warp);

		if (!m_param.fusion_enable_nonRigidSolver)
			return;

#ifdef ENABLE_CPU_DEBUG
		CpuGaussNewton solver;
		solver.init(m_warpField, m_vmap_cano, m_nmap_cano, m_param, m_kinect_intr);
#else
		// icp iteration
		m_gsSolver->init(m_warpField, m_vmap_cano, m_nmap_cano, m_param, m_kinect_intr);
#endif
		float energy = FLT_MAX;
		for (int icp_iter = 0; icp_iter < m_param.fusion_nonRigidICP_maxIter; icp_iter++)
		{
#ifdef ENABLE_CPU_DEBUG
			solver.findCorr(m_vmap_curr_pyd[0], m_nmap_curr_pyd[0], m_vmap_warp, m_nmap_warp);
			solver.solve(false);
#else
			// Gauss-Newton Optimization, findding correspondence internal
			float oldEnergy = energy, data_energy=0.f, reg_energy=0.f;
			energy = m_gsSolver->solve(m_vmap_curr_pyd[0], m_nmap_curr_pyd[0], 
				m_vmap_warp, m_nmap_warp, &data_energy, &reg_energy);

			//printf("icp, energy(data,reg): %d %f = %f + %f\n", icp_iter, energy, data_energy, reg_energy);
			if (energy > oldEnergy)
				break;

			// update the warp field
			m_gsSolver->updateWarpField();
#endif

			//// update warped mesh and render for visiblity
			//if (icp_iter < m_param.fusion_nonRigidICP_maxIter - 1)
			//{
			//	m_warpField->warp(*m_canoMesh, *m_warpedMesh);
			//	m_warpedMesh->renderToCanonicalMaps(*m_camera, m_canoMesh, m_vmap_cano, m_nmap_cano);
			//}
			m_warpField->warp(m_vmap_cano, m_nmap_cano, m_vmap_warp, m_nmap_warp);
#ifndef ENABLE_CPU_DEBUG
			if (m_param.fusion_post_rigid_factor)
				m_gsSolver->factor_out_rigid();
#endif
		}// end for icp_iter

		// finally, re-factor out the rigid part across all nodes

	}

	void DynamicFusionProcessor::nonRigidTsdfFusion()
	{
		fusion();

#ifdef SPARSE_VOLUME_TESTING
		cudaSafeCall(cudaThreadSynchronize(), "1");
		ldp::tic();
		VoxelBlockAllocation(m_depth_input);
		cudaSafeCall(cudaThreadSynchronize(), "2");
		VisibleVoxelBlockSelection();
		cudaSafeCall(cudaThreadSynchronize(), "3");
		printf("%d %d %d %d\n", m_sparseVolume->voxel_block_number_,
			m_sparseVolume->selected_voxel_block_number_,
			m_sparseVolume->hash_entry_size_,
			m_sparseVolume->selected_hash_entry_number_);
		VoxelBlockUpdate(m_depth_input);
		cudaSafeCall(cudaThreadSynchronize(), "4");
		ldp::toc("sparse fusion");
#endif
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
#ifdef ENABLE_BILATERAL_PRE_FILTER
		bilateralFilter(m_depth_input, m_depth_curr_pyd[0]);
#else
		m_depth_input.copyTo(m_depth_curr_pyd[0]);
#endif
		createVMap(m_kinect_intr(0), m_depth_curr_pyd[0], m_vmap_curr_pyd[0]);
		createNMap(m_vmap_curr_pyd[0], m_nmap_curr_pyd[0]);

		//// debug
		//static int a = 0;
		//if (a++ <= 100)
		//{
		//	std::vector<float4> tmp(KINECT_WIDTH*KINECT_HEIGHT);
		//	m_vmap_curr_pyd[0].download(tmp.data(), KINECT_WIDTH*sizeof(float4));
		//	ObjMesh mesh;
		//	for (int i = 0; i < tmp.size(); i++)
		//		mesh.vertex_list.push_back(ldp::Float3(tmp[i].x, tmp[i].y, tmp[i].z));
		//	mesh.saveObj(("D:/" + std::to_string(a) + ".obj").c_str());
		//	system("pause");
		//}
		//// end debug

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
		else if(!m_param.fusion_enable_rigidSolver)// ldp debug
			return m_warpField->get_rigidTransform();

		// now estimate rigid transform
		Tbx::Transfo c2v = m_warpField->get_rigidTransform().fast_invert();
		Tbx::Mat3	c2v_Rprev = c2v.get_mat3();
		Tbx::Vec3	c2v_tprev = c2v.get_translation();

		Tbx::Mat3	c2v_Rcurr = c2v_Rprev;
		Tbx::Vec3	c2v_tcurr = c2v_tprev;
		
		const int icp_iterations[] = { m_param.fusion_rigid_ICP_iter[2], 
			m_param.fusion_rigid_ICP_iter[1],
			m_param.fusion_rigid_ICP_iter[0] };
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

	//===============================Sparse Volume Testing=========================================
#ifdef SPARSE_VOLUME_TESTING
	void DynamicFusionProcessor::VoxelBlockAllocation(const DepthMap& depth_float_frame_d)
	{
		//	hen a new frame arrives, we insure that all the voxels 
		// within truncation distance of the depth pixel are allocated
		Tbx::Transfo c2w = m_warpField->get_rigidTransform().fast_invert();
		Mat33  device_Rc2w = convert(c2w.get_mat3());
		float3 device_tc2w = convert(c2w.get_translation());

		int3	chunk_dim;
		chunk_dim.x = CHUNK_DIM_X, chunk_dim.y = CHUNK_DIM_Y, chunk_dim.z = CHUNK_DIM_Z;
		float3	chunk_min_xyz;
		chunk_min_xyz.x = CHUNK_MIN_X, chunk_min_xyz.y = CHUNK_MIN_Y, chunk_min_xyz.z = CHUNK_MIN_Z;

		allocVoxelBlock(
			depth_float_frame_d,
			m_kinect_intr,
			device_Rc2w,
			device_tc2w,
			m_sparseVolume->getBlockSize(),
			m_sparseVolume->getTsdfTruncDist(),
			m_sparseVolume->hash_entry_,
			BUCKET_SIZE,
			m_sparseVolume->hash_bucket_atomic_lock_,
			m_sparseVolume->voxel_block_,
			m_sparseVolume->available_voxel_block_,
			m_sparseVolume->hash_parameters_,
			m_sparseVolume->voxel_block_number_,
			chunk_dim,
			chunk_min_xyz,
			CHUNK_SIZE,
			m_sparseVolume->chunk_on_CPU_);
	}

	void DynamicFusionProcessor::VisibleVoxelBlockSelection()
	{
		Tbx::Transfo w2c = m_warpField->get_rigidTransform();

		Mat33  device_Rw2c = convert(w2c.get_mat3());
		float3 device_tw2c = convert(w2c.get_translation());

		selectVisibleHashEntry(
			m_sparseVolume->hash_entry_scan_,
			m_sparseVolume->selected_hash_entry_number_,
			m_kinect_intr,
			dfusion::KINECT_WIDTH,
			dfusion::KINECT_HEIGHT,
			0.3f,
			8.0f,
			device_Rw2c,
			device_tw2c,
			m_sparseVolume->getBlockSize(),
			m_sparseVolume->hash_entry_,
			m_sparseVolume->selected_hash_entry_,
			nullptr);

		m_sparseVolume->selected_voxel_block_number_ = m_sparseVolume->selected_hash_entry_number_;
	}

	void DynamicFusionProcessor::VoxelBlockUpdate(const DepthMap& depth_float_frame_d)
	{
		Tbx::Transfo w2c = m_warpField->get_rigidTransform();

		Mat33  device_Rw2c = convert(w2c.get_mat3());
		float3 device_tw2c = convert(w2c.get_translation());

		updateVoxelBlock(
			depth_float_frame_d,
			m_kinect_intr,
			device_Rw2c,
			device_tw2c,
			m_sparseVolume->getBlockSize(),
			m_sparseVolume->getVoxelSize(),
			m_sparseVolume->getTsdfTruncDist(),
			m_sparseVolume->hash_entry_,
			BUCKET_SIZE,
			m_sparseVolume->hash_bucket_atomic_lock_,
			m_sparseVolume->voxel_block_,
			m_sparseVolume->available_voxel_block_,
			m_sparseVolume->hash_parameters_,
			m_sparseVolume->selected_hash_entry_,
			m_sparseVolume->selected_hash_entry_number_,
			m_sparseVolume->delete_hash_entry_,
			0.99f);
	}

	void DynamicFusionProcessor::OutActiveRegionVoxelBlockSelection()
	{
		Tbx::Transfo c2w = m_warpField->get_rigidTransform().fast_invert();
		Tbx::Point3	active_region_center(0, 0, m_sparseVolume->active_region_offset_);
		active_region_center = c2w * active_region_center;

		float3	center, chunk_min_xyz = convert(active_region_center);
		int3	chunk_dim;
		chunk_min_xyz.x = CHUNK_MIN_X;
		chunk_min_xyz.y = CHUNK_MIN_Y;
		chunk_min_xyz.z = CHUNK_MIN_Z;
		chunk_dim.x = CHUNK_DIM_X;
		chunk_dim.y = CHUNK_DIM_Y;
		chunk_dim.z = CHUNK_DIM_Z;

		selectOutActiveRegionHashEntry(
			m_sparseVolume->hash_entry_scan_,
			m_sparseVolume->selected_hash_entry_number_,
			center,
			m_sparseVolume->active_region_radius_,
			chunk_min_xyz,
			chunk_dim,
			m_sparseVolume->getBlockSize(),
			m_sparseVolume->hash_entry_,
			m_sparseVolume->selected_hash_entry_);
	}

	int DynamicFusionProcessor::GPU2HostStreaming(){
		//	select all entry out of active region
		OutActiveRegionVoxelBlockSelection();

		//	copy all selected entry and corresponding voxel blocks to buffer
		m_sparseVolume->selected_hash_entry_number_ = std::min(
			m_sparseVolume->selected_hash_entry_number_, BUFFER_SIZE);
		if (m_sparseVolume->selected_hash_entry_number_)
		{
			int3	chunk_dim;
			chunk_dim.x = CHUNK_DIM_X, chunk_dim.y = CHUNK_DIM_Y, chunk_dim.z = CHUNK_DIM_Z;
			float3	chunk_min_xyz;
			chunk_min_xyz.x = CHUNK_MIN_X, chunk_min_xyz.y = CHUNK_MIN_Y, chunk_min_xyz.z = CHUNK_MIN_Z;

			streamGPU2Buffer(
				m_sparseVolume->hash_entry_,
				BUCKET_SIZE,
				m_sparseVolume->hash_bucket_atomic_lock_,
				m_sparseVolume->voxel_block_,
				m_sparseVolume->voxel_block_buffer_,
				m_sparseVolume->available_voxel_block_,
				m_sparseVolume->hash_parameters_,
				m_sparseVolume->selected_hash_entry_,
				m_sparseVolume->selected_hash_entry_number_,
				chunk_dim,
				chunk_min_xyz,
				CHUNK_SIZE,
				m_sparseVolume->getBlockSize(),
				m_sparseVolume->chunk_on_CPU_);
			//std::cout << "streamed out: " << selected_hash_entry_number << std::endl;
		}

		//	transfer buffer to host
		if (m_sparseVolume->selected_hash_entry_number_)
		{
			cudaMemcpy(m_sparseVolume->selected_hash_entry_h_.data(), m_sparseVolume->selected_hash_entry_.ptr(), 
				sizeof(HashEntry)*m_sparseVolume->selected_hash_entry_number_, cudaMemcpyDeviceToHost);
			cudaMemcpy(m_sparseVolume->voxel_block_buffer_h_.data(), m_sparseVolume->voxel_block_buffer_.ptr(),
				sizeof(VoxelBlock)*m_sparseVolume->selected_hash_entry_number_, cudaMemcpyDeviceToHost);
		}

		//	insert voxel blocks into corresponding chunk
		for (int i = 0; i<m_sparseVolume->selected_hash_entry_number_; i++){
			//	calculate chunk id
			ldp::Int3 block_grid(
				m_sparseVolume->selected_hash_entry_h_[i].position[0],
				m_sparseVolume->selected_hash_entry_h_[i].position[1],
				m_sparseVolume->selected_hash_entry_h_[i].position[2]);
			ldp::Float3 block_xyz = block_grid * m_sparseVolume->getBlockSize() + ldp::Float3(1e-4f, 1e-4f, 1e-4f);
			ldp::Float3 chunk_min_xyz(CHUNK_MIN_X, CHUNK_MIN_Y, CHUNK_MIN_Z);
			ldp::Int3 chunk_grid = (block_xyz - chunk_min_xyz) / CHUNK_SIZE;
			int chunk_idx = (chunk_grid.z * CHUNK_DIM_Y + chunk_grid.y) * CHUNK_DIM_X + chunk_grid.x;

			if (!m_sparseVolume->chunk_[chunk_idx]){
				m_sparseVolume->chunk_[chunk_idx] = new Chunk;
			}

			m_sparseVolume->chunk_[chunk_idx]->AddData(m_sparseVolume->selected_hash_entry_h_[i], 
				m_sparseVolume->voxel_block_buffer_h_[i]);
		}

		return m_sparseVolume->selected_hash_entry_number_;
	}

	int DynamicFusionProcessor::Host2GPUStreaming(){
		yz::Matrix4x4f c2w = world2camera_record.back().Inverse();
		yz::Vec3f	active_region_center(0, 0, active_region_offset);
		active_region_center = c2w.TransformVector(active_region_center);

		yz::Vec3f chunk_min_xyz(CHUNK_MIN_X, CHUNK_MIN_Y, CHUNK_MIN_Z);
		int chunk_to_stream = -1;
		float dist;

		//	get lower and upper limit of chunks to check distance
		yz::Vec3f r = active_region_center - chunk_min_xyz;
		int chunk_x_min = yz::myMax(0, (r.x - active_region_radius) / CHUNK_SIZE);
		int chunk_x_max = yz::myMin(CHUNK_DIM_X - 1, (r.x + active_region_radius) / CHUNK_SIZE);
		int chunk_y_min = yz::myMax(0, (r.y - active_region_radius) / CHUNK_SIZE);
		int chunk_y_max = yz::myMin(CHUNK_DIM_Y - 1, (r.y + active_region_radius) / CHUNK_SIZE);
		int chunk_z_min = yz::myMax(0, (r.z - active_region_radius) / CHUNK_SIZE);
		int chunk_z_max = yz::myMin(CHUNK_DIM_Z - 1, (r.z + active_region_radius) / CHUNK_SIZE);

		//	find the chunk whose furthest point is most close to active_region_center
		for (int k = chunk_z_min; k <= chunk_z_max; k++){
			for (int j = chunk_y_min; j <= chunk_y_max; j++){
				for (int i = chunk_x_min; i <= chunk_x_max; i++){
					int idx = (k * CHUNK_DIM_Y + j) * CHUNK_DIM_X + i;
					if (!chunk[idx])	//	this chunk is empty, skip it
						continue;

					yz::Vec3f aabb_min = chunk_min_xyz + yz::Vec3f(i, j, k) * CHUNK_SIZE;
					yz::Vec3f corner[8] = {
						aabb_min + yz::Vec3f(0, 0, 0) * CHUNK_SIZE,
						aabb_min + yz::Vec3f(1, 0, 0) * CHUNK_SIZE,
						aabb_min + yz::Vec3f(0, 1, 0) * CHUNK_SIZE,
						aabb_min + yz::Vec3f(1, 1, 0) * CHUNK_SIZE,
						aabb_min + yz::Vec3f(0, 0, 1) * CHUNK_SIZE,
						aabb_min + yz::Vec3f(1, 0, 1) * CHUNK_SIZE,
						aabb_min + yz::Vec3f(0, 1, 1) * CHUNK_SIZE,
						aabb_min + yz::Vec3f(1, 1, 1) * CHUNK_SIZE };
					float far_dist = 0;
					for (int ii = 0; ii < 8; ii++){
						float len = (corner[ii] - active_region_center).Length();
						if (len > far_dist)
							far_dist = len;
					}
					if (far_dist < active_region_radius){// this chunk is completely inside the active sphere
						if (chunk_to_stream == -1){
							chunk_to_stream = idx;
							dist = far_dist;
						}
						else if (far_dist < dist){
							chunk_to_stream = idx;
							dist = far_dist;
						}
					}
				}
			}
		}

		//	if no chunk to stream, just return
		if (chunk_to_stream == -1)
			return chunk_to_stream;

		//	setup the chunk buffer on host
		int block_num = chunk[chunk_to_stream]->hash_entry.size();
		int i = 0;
		for (std::list<VoxelBlock>::iterator iter = chunk[chunk_to_stream]->voxel_block.begin();
			iter != chunk[chunk_to_stream]->voxel_block.end();
			iter++, i++)
		{
			selected_hash_entry_h[i] = chunk[chunk_to_stream]->hash_entry[i];
			voxel_block_buffer_h[i] = *iter;
		}
		if (i != block_num){
			std::cout << "error: KinectFusionReconstruction::Host2GPUStreaming, size don't match" << std::endl;
		}

		//	delete the chunk on CPU
		delete chunk[chunk_to_stream];
		chunk[chunk_to_stream] = NULL;

		//	transfer host to buffer
		if (block_num){
			cpyh2d(selected_hash_entry.ptr(), &selected_hash_entry_h[0], sizeof(HashEntry)*block_num);
			cpyh2d(voxel_block_buffer.ptr(), &voxel_block_buffer_h[0], sizeof(VoxelBlock)*block_num);
		}

		if (block_num){
			selected_hash_entry_number = block_num;
			streamBuffer2GPU(
				hash_entry,
				BUCKET_SIZE,
				hash_bucket_atomic_lock,
				voxel_block,
				voxel_block_buffer,
				available_voxel_block,
				hash_parameters,
				selected_hash_entry,
				selected_hash_entry_number,
				chunk_on_CPU,
				chunk_to_stream);
			//std::cout << "streamed in: " << selected_hash_entry_number << std::endl;
		}

		return chunk_to_stream;
	}
#endif
}