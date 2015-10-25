#pragma once

#include <QtOpenGL>
#include "Camera.h"
#include "kinect_util.h"
class DepthViewer : public QGLWidget
{
	Q_OBJECT

public:
	DepthViewer(QWidget *parent);
	~DepthViewer();

	void setImage_h(const dfusion::depthtype* image_h, int w, int h);
	void setImage_d(PtrStepSz<dfusion::depthtype> image_d);
	void setNormal_d(const dfusion::MapArr& image_d);

	void download_currentmap(std::vector<uchar4>& hostmap);

	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();
protected:
	QGLFunctions *m_gl_func;
	GLuint m_texture_id;
	Camera m_camera;
	std::vector<ldp::UChar4> m_depthColors_h;

	PtrStepSz<uchar4> m_pbo_buffer;
	GLuint m_pbo_id;
	cudaGraphicsResource* m_pbo_cuda_res;
};