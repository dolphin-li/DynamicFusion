#include "dynamicfusionui.h"
#include "global_data_holder.h"
DynamicFusionUI::DynamicFusionUI(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	setAcceptDrops(true);
	m_state = DynamicFusionUI::Live;
	m_frameIndex = 0;
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
	setWindowTitle(QString().sprintf("FPS:%.1f", fps));
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
		m_state = DynamicFusionUI::Loading;
		m_frameIndex = 0;
	}
	// if dropped a file, then we load as a dumpped volume.
	else if (info.isFile())
	{

	}
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
		m_state = DynamicFusionUI::Pause;
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
#if 1
	Camera cam;
	ui.widgetWarpedView->getCameraInfo(cam);
	cam.setViewPort(0, dfusion::KINECT_WIDTH, 0, dfusion::KINECT_HEIGHT);
	g_dataholder.m_rayCaster.setCamera(cam);

	g_dataholder.m_rayCaster.shading(g_dataholder.m_lights,
		g_dataholder.m_warpedview_shading, 
		g_dataholder.m_view_normalmap);

	ui.widgetWarpedView->setRayCastingShadingImage(g_dataholder.m_warpedview_shading);
#else

#if 0
	g_dataholder.m_marchCube.run(g_dataholder.m_mesh);
	///debug1
	static int a = 0;
	if (a == 0)
	{
		a = 1;
		ObjMesh mesh;
		g_dataholder.meshCopy(g_dataholder.m_mesh, mesh);
		mesh.saveObj("D:/1.obj");
	}
#endif
#endif
}

void DynamicFusionUI::on_actionSave_triggered()
{

}
void DynamicFusionUI::on_actionLoad_triggered()
{

}
void DynamicFusionUI::on_actionLoad_frames_triggered()
{
	State lastState = m_state;
	m_state = DynamicFusionUI::Pause;
	m_currentPath = QFileDialog::getExistingDirectory(this, "load folder");
	if (m_currentPath != "")
	{
		m_frameIndex = 0;
		m_state = DynamicFusionUI::Loading;
	}
	else
		m_state = lastState;
}
void DynamicFusionUI::on_actionRecord_frames_triggered()
{
	if (ui.actionRecord_frames->isChecked())
	{
		m_state = DynamicFusionUI::Pause;
		m_currentPath = QFileDialog::getExistingDirectory(this, "record folder");
		if (m_currentPath != "")
		{
			m_frameIndex = 0;
			m_state = DynamicFusionUI::Saving;
		}
		else
		{
			m_state = DynamicFusionUI::Live;
		}
	}
	else
	{
		m_state = DynamicFusionUI::Live;
	}
}
void DynamicFusionUI::on_actionLoad_volume_triggered()
{
	State lastState = m_state;
	m_state = DynamicFusionUI::Pause;
	QString name = QFileDialog::getOpenFileName(this, "load volume", "");
	if (!name.isEmpty())
	{
		m_state = DynamicFusionUI::ShowLoadedStaticVolume;
		try
		{
			mpu::VolumeData volume;
			volume.load(name.toStdString().c_str());
			g_dataholder.m_volume.initFromHost(&volume);

			//g_dataholder.m_rayCaster.init(g_dataholder.m_volume);
			g_dataholder.m_marchCube.init(&g_dataholder.m_volume, 256, 2);

#if 0
			//debug
			g_dataholder.m_volume.download(&volume);
			volume.save((name+"_test.dvol").toStdString().c_str());
#endif


		}
		catch (std::exception e)
		{
			std::cout << e.what() << std::endl;
		}
	}
	else
		m_state = lastState;
}
