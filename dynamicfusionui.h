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
	void on_actionPause_triggered();
	void on_rbRayCasting_clicked();
	void on_rbMarchCube_clicked();
	void on_pbReset_clicked();

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

	void on_cbShowMesh_clicked();
	void on_cbShowNodes_clicked();
	void on_cbShowGraph_clicked();
	void on_cbShowCorr_clicked();
	void on_sbShowGraphLevel_valueChanged(int);
	void on_sbActiveNode_valueChanged(int v);

	void on_sbNodeRadius_valueChanged(int v);
	void on_dbDwLvScale_valueChanged(double v);
	void on_sbICPIter_valueChanged(int v);
	void on_sbGSIter_valueChanged(int v);
	void on_cbDumpFrames_clicked();
	void on_cbEnableNonRigid_clicked();
	void on_dbBeta_valueChanged(double v);
	void on_dbLambda_valueChanged(double v);

	void on_gbAutoReset_clicked();
	void on_sbAutoResetSeconds_valueChanged(int v);
	void on_sbMaxWeights_valueChanged(int v);
protected:
	void updateUiFromParam();

	void frameLoading();
	void frameSaving();
	void frameLive();

	void updateLoadedStaticVolume();
	void updateDynamicFusion();

private:
	Ui::DynamicFusionUIClass ui;

	int m_fpsTimerId;
	int m_autoResetTimerId;
	double m_autoResetRemaingTime;

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
	State m_lastState;
	RenderType m_renderType;
	QString m_currentPath;
	bool m_view_normalmap;
	int m_frameIndex;
private:

	void setState(State s){ m_lastState = m_state;  m_state = s; }
	void restoreState(){ m_state = m_lastState; }
};

#endif // DYNAMICFUSIONUI_H
