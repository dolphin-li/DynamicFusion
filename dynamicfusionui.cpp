#include "dynamicfusionui.h"
#include "global_data_holder.h"
#include "WarpField.h"
DynamicFusionUI::DynamicFusionUI(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setAcceptDrops(true);
	m_frameIndex = 0;
	m_view_normalmap = false;
	m_lastState = DynamicFusionUI::Live;
	m_state = DynamicFusionUI::Pause;
	m_renderType = RenderRayCasting;
	updateUiFromParam();

	
	g_dataholder.init();

	startTimer(30);
}

DynamicFusionUI::~DynamicFusionUI()
{

}

void DynamicFusionUI::timerEvent(QTimerEvent* ev)
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

		//// visualize the depth via jet map, calculate it on GPU
		//ui.widgetDepth->setImage_h(g_dataholder.m_depth_h.data(), dfusion::KINECT_WIDTH, dfusion::KINECT_HEIGHT);
		ui.widgetDepth->setImage_d(g_dataholder.m_depth_d);

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
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
	}

	gtime_t time_e = gtime_now();
	double sec = gtime_seconds(time_s, time_e);
	double fps = 1.0 / sec;
	setWindowTitle(QString().sprintf("FPS:%.1f;  Nodes: %d %d %d %d", fps,
		g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(0),
		g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(1),
		g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(2),
		g_dataholder.m_processor.getWarpField()->getNumNodesInLevel(3)
		));
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

	ui.cbShowMesh->setChecked(g_dataholder.m_dparam.view_show_mesh);
	ui.cbShowNodes->setChecked(g_dataholder.m_dparam.view_show_nodes);
	ui.cbShowGraph->setChecked(g_dataholder.m_dparam.view_show_graph);
	ui.cbShowCorr->setChecked(g_dataholder.m_dparam.view_show_corr);

	ui.sbShowGraphLevel->setMaximum(dfusion::WarpField::GraphLevelNum);
	ui.sbShowGraphLevel->setValue(g_dataholder.m_dparam.view_show_graph_level);
	ui.sbActiveNode->setValue(g_dataholder.m_dparam.view_activeNode_id);
}

void DynamicFusionUI::frameLoading()
{
	QDir dir(m_currentPath);
	if (!dir.exists())
		throw std::exception(("error input path:" + m_currentPath.toStdString()).c_str());
	QString name = dir.absoluteFilePath(QString().sprintf("%08d.depth", m_frameIndex++));

	try
	{
		g_dataholder.loadDepth(g_dataholder.m_depth_h, name.toStdString());
		g_dataholder.m_depth_d.upload(g_dataholder.m_depth_h.data(), dfusion::KINECT_WIDTH*sizeof(dfusion::depthtype),
			dfusion::KINECT_HEIGHT, dfusion::KINECT_WIDTH);
	}
	catch (std::exception e)
	{
		std::cout << e.what() << std::endl;
		setState(DynamicFusionUI::Pause);
	}
}

void DynamicFusionUI::frameSaving()
{
	QDir dir(m_currentPath);
	if (!dir.exists())
		mkdir(m_currentPath.toStdString());
	QString name = dir.absoluteFilePath(QString().sprintf("%08d.depth", m_frameIndex++));

	g_dataholder.saveDepth(g_dataholder.m_depth_h, name.toStdString());
	printf("saved: %s\n", name.toStdString().c_str());
}

void DynamicFusionUI::frameLive()
{
	g_dataholder.m_kinect.GetDepthColorIntoBuffer(g_dataholder.m_depth_h.data(), nullptr);
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

	// debug, save rendered image
#if 1
	// warp view
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
	name.sprintf("data/screenshots/%06d.png", fid);
	img.save(name);

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
	name1.sprintf("data/screenshots/e_%06d.png", fid);
	img.save(name1);
#endif
	// end debug


}

void DynamicFusionUI::on_actionPause_triggered()
{
	if (ui.actionPause->isChecked())
		setState(DynamicFusionUI::Pause);
	else
		restoreState();
}

void DynamicFusionUI::on_pbReset_clicked()
{
	g_dataholder.m_processor.reset();
}

void DynamicFusionUI::on_actionSave_triggered()
{

}
void DynamicFusionUI::on_actionLoad_triggered()
{

}
void DynamicFusionUI::on_actionLoad_frames_triggered()
{
	setState(DynamicFusionUI::Pause);
	m_currentPath = QFileDialog::getExistingDirectory(this, "load folder", "data/frames/");
	if (m_currentPath != "")
	{
		m_frameIndex = 0;
		setState(DynamicFusionUI::Loading);
		g_dataholder.m_processor.reset();
	}
	else
		restoreState();
}
void DynamicFusionUI::on_actionRecord_frames_triggered()
{
	if (ui.actionRecord_frames->isChecked())
	{
		setState(DynamicFusionUI::Pause);
		m_currentPath = QFileDialog::getExistingDirectory(this, "record folder", "data/frames/");
		if (m_currentPath != "")
		{
			m_frameIndex = 0;
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
		g_dataholder.m_dparam.voxels_per_meter = v;
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
