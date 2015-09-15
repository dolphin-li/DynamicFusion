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
protected:
	void frameLoading();
	void frameSaving();
	void frameLive();
private:
	Ui::DynamicFusionUIClass ui;

	enum State
	{
		Pause,
		Loading,
		Saving,
		Live,
	};
	State m_state;
	QString m_currentPath;
	int m_frameIndex;
};

#endif // DYNAMICFUSIONUI_H
