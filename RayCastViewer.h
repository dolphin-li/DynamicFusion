#pragma once

#include <QtOpenGL>
#include "Camera.h"
#include "definations.h"
class RayCastViewer : public QGLWidget
{
	Q_OBJECT
public:
	RayCastViewer(QWidget *parent);
	~RayCastViewer();

	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();

	void setId(int id){ m_id = id; }

	void setRayCastingShadingImage(const dfusion::ColorMap& img);

	void setSameView(const RayCastViewer* other);

	const int getId()const{ return m_id; }

	void getCameraInfo(Camera& cam)const;
	void setCameraInfo(const Camera& cam);
protected:
	void mousePressEvent(QMouseEvent *);
	void mouseReleaseEvent(QMouseEvent *);
	void mouseMoveEvent(QMouseEvent*);
	void wheelEvent(QWheelEvent*);
	void keyPressEvent(QKeyEvent*);

	void renderRayCasting();
protected:
	// camera related
	int m_id;
	Camera m_camera;
	ldp::Float3 m_defaultCameraLocation;
	ldp::Float3 m_defaultCameraDirection;
	ldp::Float3 m_defaultCameraUp;
	Qt::MouseButtons m_buttons;
	QPoint m_lastPos, m_lastMousePressPos;
	ldp::Mat4f m_rootTrans, m_defaultRootTrans;
	float m_dataScale;

	QGLFunctions* m_gl_func;
	GLuint m_texture_id;
	PtrStepSz<uchar4> m_pbo_buffer;
	GLuint m_pbo_id;
	cudaGraphicsResource* m_pbo_cuda_res;
};