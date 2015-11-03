#include "dynamicfusionui.h"
#include "global_data_holder.h"
#include "WarpField.h"
#include "GpuGaussNewtonSolver.h"

QMutex* g_mutex;
class SaveThread : public QThread
{
public:
	SaveThread() :QThread()
	{
		m_depths.reserve(g_dataholder.m_dparam.fusion_dumping_max_frame);
		m_ids.reserve(g_dataholder.m_dparam.fusion_dumping_max_frame);
	}
	void set_path(QString path)
	{
		QMutexLocker locker(g_mutex);
		m_currentPath = path;
		QDir dir(m_currentPath);
		if (!dir.exists())
			mkdir(m_currentPath.toStdString());
		g_dataholder.m_dparam.save(fullfile(m_currentPath.toStdString(), "_param.param.txt").c_str());
	}
	void push_depth(const std::vector<dfusion::depthtype>& depth, int id)
	{
		QMutexLocker locker(g_mutex);
		m_depths.push_back(depth);
		m_ids.push_back(id);
	}
protected:
	void run()
	{
		while (1)
		{
			if (!m_depths.empty())
			{
				int id = 0;
				const std::vector<dfusion::depthtype>* depth = nullptr;
				{
					QMutexLocker locker(g_mutex);
					id = m_ids.front();
					depth = &m_depths.front();
				}

				QDir dir(m_currentPath);
				QString name = dir.absoluteFilePath(QString().sprintf("%08d.depth", id));
				g_dataholder.saveDepth(*depth, name.toStdString());
				printf("saved: %s\n", name.toStdString().c_str());

				{
					QMutexLocker locker(g_mutex);
					m_ids.pop_front();
					m_depths.pop_front();
				}
			}
		}
	}
private:
	QQueue<std::vector<dfusion::depthtype>> m_depths;
	QQueue<int> m_ids;
	QString m_currentPath;
};
SaveThread g_saveThread;

DynamicFusionUI::DynamicFusionUI(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setAcceptDrops(true);
	m_frameIndex = 0;
	m_view_normalmap = false;
	m_lastState = DynamicFusionUI::Live;
	m_state = DynamicFusionUI::Live;
	if (g_dataholder.m_dparam.fusion_loading_mode)
		m_state = DynamicFusionUI::Pause;
	m_renderType = RenderRayCasting;
	updateUiFromParam();

	
	g_dataholder.init();

	m_fpsTimerId = startTimer(30);
	m_autoResetTimerId = startTimer(g_dataholder.m_dparam.view_autoreset_seconds*1000);
	m_autoResetRemaingTime = g_dataholder.m_dparam.view_autoreset_seconds;

	g_mutex = new QMutex();
	g_saveThread.start();
}

DynamicFusionUI::~DynamicFusionUI()
{
	g_saveThread.terminate();
	delete g_mutex;
}

void DynamicFusionUI::timerEvent(QTimerEvent* ev)
{
	if (m_fpsTimerId == ev->timerId())
	{
		gtime_t time_s = gtime_now();

		try
		{
			//// process a new kinect frame.
			switch (m_state)
			{
			case DynamicFusionUI::Loading:
				frameLoading();
				break;
			case DynamicFusionUI::Saving:
				frameLive();
				frameSaving();
				break;
			case DynamicFusionUI::Live:
				frameLive();
				break;
			default:
				break;
			}

			//// process viewers
			switch (m_state)
			{
			case DynamicFusionUI::ShowLoadedStaticVolume:
				updateLoadedStaticVolume();
				break;
			case DynamicFusionUI::Live:
			case DynamicFusionUI::Loading:
			case DynamicFusionUI::Pause:
				updateDynamicFusion();
				break;
			default:
				break;
			}

			//// visualize the depth via jet map, calculate it on GPU
			//ui.widgetDepth->setImage_h(g_dataholder.m_depth_h.data(), 
			// dfusion::KINECT_WIDTH, dfusion::KINECT_HEIGHT);
			if (g_dataholder.m_processor.hasRawDepth() && m_state != Saving)
			{
				const dfusion::MapArr& nmap = g_dataholder.m_processor.getRawDepthNormal();
				ui.widgetDepth->setNormal_d(nmap);
			}
			else
				ui.widgetDepth->setImage_d(g_dataholder.m_depth_d);
		}
		catch (std::exception e)
		{
			std::cout << e.what() << std::endl;
		}

		gtime_t time_e = gtime_now();
		double sec = gtime_seconds(time_s, time_e);
		double fps = 1.0 / sec;
		m_autoResetRemaingTime -= sec;
		setWindowTitle(QString().sprintf("[%d] FPS:%.1f;  Nodes: %d %d %d %d; Reset: %.1f", 
			m_frameIndex, fps,
			g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(0),
			g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(1),
			g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(2),
			g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(3),
			m_autoResetRemaingTime
			));
	}// end if fps timer id
	else if (m_autoResetTimerId == ev->timerId())
	{
		m_autoResetRemaingTime = g_dataholder.m_dparam.view_autoreset_seconds;
		if (g_dataholder.m_dparam.view_autoreset)
		{
			g_dataholder.m_processor.reset();
			printf("auto reset!");
		}
	}
}

void DynamicFusionUI::dragEnterEvent(QDragEnterEvent* ev)
{
	ev->acceptProposedAction();
}

void DynamicFusionUI::dropEvent(QDropEvent *ev)
{
	QString s = ev->mimeData()->text();
	QUrl url = QUrl(s);
	QString path = url.toLocalFile();

	QFile file(path);
	QFileInfo info(file);

	// if dropped a directory, then we load all frames in it.
	if (info.isDir())
	{
		m_currentPath = path;
		setState(DynamicFusionUI::Loading);
		m_frameIndex = 0;
	}
	// if dropped a file, then we load as a dumpped volume.
	else if (info.isFile())
	{

	}
}

void DynamicFusionUI::updateUiFromParam()
{
	ui.rbMarchCube->setChecked(m_renderType == RenderMarchCube);
	ui.rbRayCasting->setChecked(m_renderType == RenderRayCasting);
	
	if (g_dataholder.m_dparam.volume_resolution[0] == 128)
		ui.rbResX128->setChecked(true);
	if (g_dataholder.m_dparam.volume_resolution[0] == 256)
		ui.rbResX256->setChecked(true);
	if (g_dataholder.m_dparam.volume_resolution[0] == 384)
		ui.rbResX384->setChecked(true);
	if (g_dataholder.m_dparam.volume_resolution[0] == 512)
		ui.rbResX512->setChecked(true);
	if (g_dataholder.m_dparam.volume_resolution[1] == 128)
		ui.rbResY128->setChecked(true);
	if (g_dataholder.m_dparam.volume_resolution[1] == 256)
		ui.rbResY256->setChecked(true);
	if (g_dataholder.m_dparam.volume_resolution[1] == 384)
		ui.rbResY384->setChecked(true);
	if (g_dataholder.m_dparam.volume_resolution[1] == 512)
		ui.rbResY512->setChecked(true);
	if (g_dataholder.m_dparam.volume_resolution[2] == 128)
		ui.rbResZ128->setChecked(true);
	if (g_dataholder.m_dparam.volume_resolution[2] == 256)
		ui.rbResZ256->setChecked(true);
	if (g_dataholder.m_dparam.volume_resolution[2] == 384)
		ui.rbResZ384->setChecked(true);
	if (g_dataholder.m_dparam.volume_resolution[2] == 512)
		ui.rbResZ512->setChecked(true);
	ui.sbVoxelsPerMeter->setValue(g_dataholder.m_dparam.voxels_per_meter);

	ui.cbNoRigid->setChecked(g_dataholder.m_dparam.view_no_rigid);
	ui.cbShowMesh->setChecked(g_dataholder.m_dparam.view_show_mesh);
	ui.cbShowNodes->setChecked(g_dataholder.m_dparam.view_show_nodes);
	ui.cbShowGraph->setChecked(g_dataholder.m_dparam.view_show_graph);
	ui.cbShowCorr->setChecked(g_dataholder.m_dparam.view_show_corr);

	ui.sbShowGraphLevel->setMaximum(dfusion::WarpField::GraphLevelNum);
	ui.sbShowGraphLevel->setValue(g_dataholder.m_dparam.view_show_graph_level);
	ui.sbActiveNode->setValue(g_dataholder.m_dparam.view_activeNode_id);

	ui.sbNodeRadius->setValue(g_dataholder.m_dparam.warp_radius_search_epsilon*1000);
	ui.dbDwLvScale->setValue(g_dataholder.m_dparam.warp_param_dw_lvup_scale);
	ui.dbDwSoft->setValue(g_dataholder.m_dparam.warp_param_softness);
	ui.sbICPIter->setValue(g_dataholder.m_dparam.fusion_nonRigidICP_maxIter);
	ui.sbGSIter->setValue(g_dataholder.m_dparam.fusion_GaussNewton_maxIter);
	ui.cbDumpFrames->setChecked(g_dataholder.m_dparam.fusion_dumping_each_frame);
	ui.cbEnableNonRigid->setChecked(g_dataholder.m_dparam.fusion_enable_nonRigidSolver);
	ui.cbEnableRigid->setChecked(g_dataholder.m_dparam.fusion_enable_rigidSolver);
	ui.dbBeta->setValue(g_dataholder.m_dparam.warp_radius_search_beta);
	ui.dbLambda->setValue(g_dataholder.m_dparam.fusion_lambda);

	ui.gbAutoReset->setChecked(g_dataholder.m_dparam.view_autoreset);
	ui.sbAutoResetSeconds->setValue(g_dataholder.m_dparam.view_autoreset_seconds);
	ui.sbMaxWeights->setValue(g_dataholder.m_dparam.fusion_max_weight);
	ui.dbGSStep->setValue(g_dataholder.m_dparam.fusion_GaussNewton_fixedStep);

	ui.sbFrmIdxPlus->setValue(g_dataholder.m_dparam.load_frameIndx_plus_num);
}

void DynamicFusionUI::frameLoading()
{
	QDir dir(m_currentPath);
	if (!dir.exists())
		throw std::exception(("error input path:" + m_currentPath.toStdString()).c_str());
	QString name = dir.absoluteFilePath(QString().sprintf("%08d.depth", m_frameIndex));
	m_frameIndex += g_dataholder.m_dparam.load_frameIndx_plus_num;

	try
	{
		g_dataholder.loadDepth(g_dataholder.m_depth_h, name.toStdString());
		g_dataholder.m_depth_d.upload(g_dataholder.m_depth_h.data(), dfusion::KINECT_WIDTH*sizeof(dfusion::depthtype),
			dfusion::KINECT_HEIGHT, dfusion::KINECT_WIDTH);
	}
	catch (std::exception e)
	{
		setState(Pause);
		std::cout << e.what() << std::endl;
	}
}

void DynamicFusionUI::frameSaving()
{
	g_saveThread.push_depth(g_dataholder.m_depth_h, m_frameIndex++);
}

void DynamicFusionUI::frameLive()
{
	g_dataholder.m_kinect.GetDepthColorIntoBuffer(g_dataholder.m_depth_h.data(), 
		nullptr, false, g_dataholder.m_dparam.mirror_input);
	g_dataholder.m_depth_d.upload(g_dataholder.m_depth_h.data(), dfusion::KINECT_WIDTH*sizeof(dfusion::depthtype),
		dfusion::KINECT_HEIGHT, dfusion::KINECT_WIDTH);
}

void DynamicFusionUI::updateLoadedStaticVolume()
{
	Camera cam;
	ui.widgetWarpedView->getCameraInfo(cam);
	cam.setViewPort(0, dfusion::KINECT_WIDTH, 0, dfusion::KINECT_HEIGHT);
	cam.setPerspective(KINECT_DEPTH_V_FOV, float(dfusion::KINECT_WIDTH)/dfusion::KINECT_HEIGHT, 
		KINECT_NEAREST_METER, 30.f);
	g_dataholder.m_rayCaster.setCamera(cam);

	switch (m_renderType)
	{
	case DynamicFusionUI::RenderRayCasting:
		g_dataholder.m_rayCaster.shading(g_dataholder.m_lights,
			g_dataholder.m_warpedview_shading,
			m_view_normalmap);
		break;
	case DynamicFusionUI::RenderMarchCube:
		g_dataholder.m_marchCube.run(g_dataholder.m_mesh);
		g_dataholder.m_mesh.renderToImg(cam, g_dataholder.m_lights, g_dataholder.m_warpedview_shading,
			g_dataholder.m_dparam);
		break;
	default:
		break;
	}

	ui.widgetWarpedView->setRayCastingShadingImage(g_dataholder.m_warpedview_shading);
}

void DynamicFusionUI::updateDynamicFusion()
{
	if (m_state != DynamicFusionUI::Pause)
		g_dataholder.m_processor.processFrame(g_dataholder.m_depth_d);

	Camera cam;

	// warped view
	ui.widgetWarpedView->getCameraInfo(cam);
	cam.setViewPort(0, ui.widgetWarpedView->width(), 0, ui.widgetWarpedView->height());
	g_dataholder.m_processor.shading(cam, g_dataholder.m_lights, 
		g_dataholder.m_warpedview_shading, false);
	ui.widgetWarpedView->setRayCastingShadingImage(g_dataholder.m_warpedview_shading);

	// cano view
	ui.widgetCanoView->getCameraInfo(cam);
	cam.setViewPort(0, ui.widgetCanoView->width(), 0, ui.widgetCanoView->height());
	g_dataholder.m_processor.shadingCanonical(cam, g_dataholder.m_lights,
		g_dataholder.m_canoview_shading, false);
	ui.widgetCanoView->setRayCastingShadingImage(g_dataholder.m_canoview_shading);

	// err map
	g_dataholder.m_processor.shadingCurrentErrorMap(g_dataholder.m_errorMap_shading,
		g_dataholder.m_dparam.view_errorMap_range);
	ui.widgetErrMap->setRayCastingShadingImage(g_dataholder.m_errorMap_shading);

	// ldp debug, save rendered image
	if (g_dataholder.m_dparam.fusion_dumping_each_frame
		&& m_frameIndex < g_dataholder.m_dparam.fusion_dumping_max_frame
		&& m_state != Pause)
	{
		// warp view
		{
			std::vector<uchar4> tmpMap(g_dataholder.m_warpedview_shading.rows()
			*g_dataholder.m_warpedview_shading.cols());
			g_dataholder.m_warpedview_shading.download(tmpMap.data(),
				g_dataholder.m_warpedview_shading.cols()*sizeof(uchar4));
			QImage img = QImage(g_dataholder.m_warpedview_shading.cols(),
				g_dataholder.m_warpedview_shading.rows(), QImage::Format::Format_ARGB32);
			for (int y = 0; y < g_dataholder.m_warpedview_shading.rows(); y++)
			for (int x = 0; x < g_dataholder.m_warpedview_shading.cols(); x++)
			{
				uchar4 v = tmpMap[y*g_dataholder.m_warpedview_shading.cols() + x];
				img.setPixel(x, y, qRgba(v.x, v.y, v.z, v.w));
			}
			int fid = g_dataholder.m_processor.getFrameId();

			QString name;
			name.sprintf("data/screenshots/%06d_%d_%d_%d_%d.png", fid,
				g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(0),
				g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(1),
				g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(2),
				g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(3));
			img.save(name);
		}
		// raw view
		{
			std::vector<uchar4> tmpMap;
			ui.widgetDepth->download_currentmap(tmpMap);
			QImage img = QImage(dfusion::KINECT_WIDTH, dfusion::KINECT_HEIGHT, QImage::Format::Format_ARGB32);
			for (int y = 0; y < dfusion::KINECT_HEIGHT; y++)
			for (int x = 0; x < dfusion::KINECT_WIDTH; x++)
			{
				uchar4 v = tmpMap[y*dfusion::KINECT_WIDTH + x];
				img.setPixel(x, y, qRgba(v.x, v.y, v.z, v.w));
			}
			int fid = g_dataholder.m_processor.getFrameId();

			QString name;
			name.sprintf("data/screenshots/raw_%06d_%d_%d_%d_%d.png", fid,
				g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(0),
				g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(1),
				g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(2),
				g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(3));
			img.save(name);
		}
#if 0
		// error map
		img = QImage(g_dataholder.m_errorMap_shading.cols(),
			g_dataholder.m_errorMap_shading.rows(), QImage::Format::Format_ARGB32);
		g_dataholder.m_errorMap_shading.download(tmpMap.data(),
			g_dataholder.m_errorMap_shading.cols()*sizeof(uchar4));
		for (int y = 0; y < g_dataholder.m_errorMap_shading.rows(); y++)
		for (int x = 0; x < g_dataholder.m_errorMap_shading.cols(); x++)
		{
			uchar4 v = tmpMap[y*g_dataholder.m_errorMap_shading.cols() + x];
			img.setPixel(x, y, qRgba(v.x, v.y, v.z, v.w));
		}

		QString name1;
		name1.sprintf("data/screenshots/e_%06d_%d_%d_%d_%d.png", fid,
			g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(0),
			g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(1),
			g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(2),
			g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(3));
		img.save(name1);
#endif
	}
}

void DynamicFusionUI::on_actionPause_triggered()
{
	if (m_state != Pause)
		setState(DynamicFusionUI::Pause);
	else
		restoreState();
}

void DynamicFusionUI::on_pbReset_clicked()
{
	g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	g_dataholder.m_processor.reset();
}

void DynamicFusionUI::on_actionSave_triggered()
{
	try
	{
		setState(DynamicFusionUI::Pause);
		QString name = QFileDialog::getSaveFileName(this, "save", "", "*.rawvol");
		if (!name.isEmpty())
		{
			try
			{
				if (!name.endsWith(".rawvol"))
					name.append(".rawvol");
				g_dataholder.m_processor.save(name.toStdString().c_str());
			}
			catch (std::exception e)
			{
				std::cout << e.what() << std::endl;
			}
		}
		restoreState();
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}

void DynamicFusionUI::on_actionLoad_triggered()
{
	try
	{
		setState(DynamicFusionUI::Pause);
		QString name = QFileDialog::getOpenFileName(this, "load", "", "*.rawvol");
		if (!name.isEmpty())
		{
			try
			{
				g_dataholder.m_processor.load(name.toStdString().c_str());
			}
			catch (std::exception e)
			{
				std::cout << e.what() << std::endl;
			}
		}
		else
			restoreState();
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_actionLoad_frames_triggered()
{
	//setState(DynamicFusionUI::Pause);
	m_currentPath = QFileDialog::getExistingDirectory(this, "load folder", "data/frames/");
	if (m_currentPath != "")
	{
		m_frameIndex = 0;
		QDir dir(m_currentPath);
		if (!dir.exists())
			throw std::exception(("error input path:" + m_currentPath.toStdString()).c_str());

		// 0. firstly we try to load existed param files, if existed, we update params.
		try
		{
			g_dataholder.m_dparam.load(fullfile(m_currentPath.toStdString(), "_param.param.txt").c_str());
			updateUiFromParam();
			g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
		}
		catch (std::exception e)
		{
			std::cout << e.what() << std::endl;
		}

		// 1. load a pre-saved volume and then start by this frame.
		int fid = 0;
		int vol_fid = 0;
		g_dataholder.m_processor.reset();
		
		for (; vol_fid < g_dataholder.m_dparam.fusion_dumping_max_frame;)
		{
			std::string volname = fullfile(m_currentPath.toStdString(), std::to_string(vol_fid) + ".rawvol");
			if (ldp::file_exist(volname.c_str()))
			{
				g_dataholder.m_processor.load(volname.c_str());
				break;
			}
			else
				vol_fid++;
		}
		if (vol_fid < g_dataholder.m_dparam.fusion_dumping_max_frame)
			fid = vol_fid;

		// 2. find the smallest available frame
		for (; fid < g_dataholder.m_dparam.fusion_dumping_max_frame; )
		{
			try
			{
				QString name = dir.absoluteFilePath(QString().sprintf("%08d.depth", fid));
				g_dataholder.loadDepth(g_dataholder.m_depth_h, name.toStdString());
				break;
			}
			catch (std::exception e)
			{
				fid++;
			}
		}
		m_frameIndex = fid;

		setState(DynamicFusionUI::Loading);
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);

		// for convinience, if there exists pre-defined volumes, we
		// just pause to visualize it.
		if (vol_fid < g_dataholder.m_dparam.fusion_dumping_max_frame)
		{
			frameLoading();
			g_dataholder.m_processor.processFrame(g_dataholder.m_depth_d);
			setState(DynamicFusionUI::Pause);
		}
	}
	//else
	//	restoreState();
}
void DynamicFusionUI::on_actionRecord_frames_triggered()
{
	if (ui.actionRecord_frames->isChecked())
	{
		//setState(DynamicFusionUI::Pause);
		m_currentPath = QFileDialog::getExistingDirectory(this, "record folder", "data/frames/");
		if (m_currentPath != "")
		{
			m_frameIndex = 0;
			g_saveThread.set_path(m_currentPath);
			setState(DynamicFusionUI::Saving);
		}
		else
		{
			setState(DynamicFusionUI::Live);
		}
	}
	else
	{
		setState(DynamicFusionUI::Live);
	}
}
void DynamicFusionUI::on_actionLoad_volume_triggered()
{
	try
	{
		setState(DynamicFusionUI::Pause);
		QString name = QFileDialog::getOpenFileName(this, "load volume", "");
		if (!name.isEmpty())
		{
			setState(DynamicFusionUI::ShowLoadedStaticVolume);
			try
			{
				mpu::VolumeData volume;
				volume.load(name.toStdString().c_str());
				g_dataholder.m_volume.initFromHost(&volume);

				g_dataholder.m_rayCaster.init(g_dataholder.m_volume);
				g_dataholder.m_marchCube.init(&g_dataholder.m_volume,
					g_dataholder.m_dparam);

				g_dataholder.m_dparam.volume_resolution[0] = volume.getResolution()[0];
				g_dataholder.m_dparam.volume_resolution[1] = volume.getResolution()[1];
				g_dataholder.m_dparam.volume_resolution[2] = volume.getResolution()[2];
				g_dataholder.m_dparam.voxels_per_meter = std::lroundf(1.f / volume.getVoxelSize());
				updateUiFromParam();
			}
			catch (std::exception e)
			{
				std::cout << e.what() << std::endl;
			}
		}
		else
			restoreState();
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_actionSave_current_mesh_triggered()
{
	try
	{
		QString name = QFileDialog::getSaveFileName(this, "save mesh", "", ".obj");
		if (!name.isEmpty())
		{
			if (!name.endsWith(".obj"))
				name.append(".obj");
			ObjMesh mesh;
			g_dataholder.m_mesh.toObjMesh(mesh);
			mesh.saveObj(name.toStdString().c_str());
		}
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbRayCasting_clicked()
{
	m_renderType = RenderRayCasting;
}
void DynamicFusionUI::on_rbMarchCube_clicked()
{
	m_renderType = RenderMarchCube;
}

void DynamicFusionUI::on_rbResX128_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[0] = 128;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbResX256_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[0] = 256;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbResX384_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[0] = 384;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbResX512_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[0] = 512;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbResY128_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[1] = 128;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbResY256_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[1] = 256;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbResY384_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[1] = 384;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbResY512_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[1] = 512;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbResZ128_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[2] = 128;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbResZ256_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[2] = 256;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbResZ384_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[2] = 384;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_rbResZ512_clicked()
{
	try
	{
		g_dataholder.m_dparam.volume_resolution[2] = 512;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_sbVoxelsPerMeter_valueChanged(int v)
{
	try
	{
		g_dataholder.m_dparam.set_voxels_per_meter(v);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}

void DynamicFusionUI::on_cbNoRigid_clicked()
{
	try
	{
		g_dataholder.m_dparam.view_no_rigid = ui.cbNoRigid->isChecked();
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_cbShowMesh_clicked()
{
	try
	{
		g_dataholder.m_dparam.view_show_mesh = ui.cbShowMesh->isChecked();
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_cbShowNodes_clicked()
{
	try
	{
		g_dataholder.m_dparam.view_show_nodes = ui.cbShowNodes->isChecked();
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_cbShowGraph_clicked()
{
	try
	{
		g_dataholder.m_dparam.view_show_graph = ui.cbShowGraph->isChecked();
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_cbShowCorr_clicked()
{
	try
	{
		g_dataholder.m_dparam.view_show_corr = ui.cbShowCorr->isChecked();
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_sbShowGraphLevel_valueChanged(int v)
{
	try
	{
		g_dataholder.m_dparam.view_show_graph_level = v;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
void DynamicFusionUI::on_sbActiveNode_valueChanged(int v)
{
	try
	{
		g_dataholder.m_dparam.view_activeNode_id = v;
		g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}

void DynamicFusionUI::on_sbNodeRadius_valueChanged(int v)
{
	g_dataholder.m_dparam.set_warp_radius_search_epsilon(float(v)/1000);
}

void DynamicFusionUI::on_dbDwLvScale_valueChanged(double v)
{
	g_dataholder.m_dparam.warp_param_dw_lvup_scale = v;
}

void DynamicFusionUI::on_dbDwSoft_valueChanged(double v)
{
	g_dataholder.m_dparam.warp_param_softness = v;
}

void DynamicFusionUI::on_sbICPIter_valueChanged(int v)
{
	g_dataholder.m_dparam.fusion_nonRigidICP_maxIter = v;
}

void DynamicFusionUI::on_sbGSIter_valueChanged(int v)
{
	g_dataholder.m_dparam.fusion_GaussNewton_maxIter = v;
}

void DynamicFusionUI::on_cbDumpFrames_clicked()
{
	g_dataholder.m_dparam.fusion_dumping_each_frame = ui.cbDumpFrames->isChecked();
}

void DynamicFusionUI::on_cbEnableNonRigid_clicked()
{
	g_dataholder.m_dparam.fusion_enable_nonRigidSolver = ui.cbEnableNonRigid->isChecked();
	g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
}

void DynamicFusionUI::on_dbBeta_valueChanged(double v)
{
	g_dataholder.m_dparam.warp_radius_search_beta = v;
}

void DynamicFusionUI::on_dbLambda_valueChanged(double v)
{
	g_dataholder.m_dparam.fusion_lambda = v;
}

void DynamicFusionUI::on_gbAutoReset_clicked()
{
	g_dataholder.m_dparam.view_autoreset = ui.gbAutoReset->isChecked();
}

void DynamicFusionUI::on_sbAutoResetSeconds_valueChanged(int v)
{
	g_dataholder.m_dparam.view_autoreset_seconds = v;
	killTimer(m_autoResetTimerId);
	m_autoResetTimerId = startTimer(g_dataholder.m_dparam.view_autoreset_seconds * 1000);
	m_autoResetRemaingTime = g_dataholder.m_dparam.view_autoreset_seconds;
}

void DynamicFusionUI::on_sbMaxWeights_valueChanged(int v)
{
	g_dataholder.m_dparam.fusion_max_weight = v;
}

void DynamicFusionUI::on_dbGSStep_valueChanged(double v)
{
	g_dataholder.m_dparam.fusion_GaussNewton_fixedStep = v;
	g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
}

void DynamicFusionUI::on_pbDebug_clicked()
{
	try
	{
		g_dataholder.m_processor.getSolver()->debug_print();
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}

void DynamicFusionUI::on_sbFrmIdxPlus_valueChanged(int v)
{
	g_dataholder.m_dparam.load_frameIndx_plus_num = v;
}

void DynamicFusionUI::on_pbUpdateParam_clicked()
{
	g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
}

void DynamicFusionUI::on_cbEnableRigid_clicked()
{
	g_dataholder.m_dparam.fusion_enable_rigidSolver = ui.cbEnableRigid->isChecked();
	g_dataholder.m_processor.updateParam(g_dataholder.m_dparam);
}
