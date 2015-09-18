#ifndef DYNAMICFUSIONUI_H
#define DYNAMICFUSIONUI_H

#include <QtWidgets/QMainWindow>
#include "ui_dynamicfusionui.h"

class DynamicFusionUI : public QMainWindow
{
	Q_OBJECT

public:
	DynamicFusionUI(QWidget *parent = 0);
	~DynamicFusionUI();

	void timerEvent(QTimerEvent* ev);
	void dragEnterEvent(QDragEnterEvent* ev);
	void dropEvent(QDropEvent *ev);

	public slots:
	void on_actionSave_triggered();
	void on_actionLoad_triggered();
	void on_actionLoad_frames_triggered();
	void on_actionRecord_frames_triggered();
	void on_actionLoad_volume_triggered();
	void on_actionSave_current_mesh_triggered();
	void on_rbRayCasting_clicked();
	void on_rbMarchCube_clicked();

	void on_rbResX128_clicked();
	void on_rbResX256_clicked();
	void on_rbResX384_clicked();
	void on_rbResX512_clicked();
	void on_rbResY128_clicked();
	void on_rbResY256_clicked();
	void on_rbResY384_clicked();
	void on_rbResY512_clicked();
	void on_rbResZ128_clicked();
	void on_rbResZ256_clicked();
	void on_rbResZ384_clicked();
	void on_rbResZ512_clicked();
	void on_sbVoxelsPerMeter_valueChanged(int v);
protected:
	void updateUiFromParam();

	void frameLoading();
	void frameSaving();
	void frameLive();

	void updateLoadedStaticVolume();
private:
	Ui::DynamicFusionUIClass ui;

	enum State
	{
		Pause,
		Loading,
		Saving,
		Live,
		ShowLoadedStaticVolume,
	};
	enum RenderType
	{
		RenderRayCasting,
		RenderMarchCube,
	};
	State m_state;
	RenderType m_renderType;
	QString m_currentPath;
	int m_frameIndex;
};

#endif // DYNAMICFUSIONUI_H
