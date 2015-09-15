#include "glew.h"
#include "DepthViewer.h"
#include "util.h"
#include <cuda_gl_interop.h>
#include <cudagl.h>

#define CHECK_GL_ERROR {\
GLenum err = glGetError();\
if (err != GL_NO_ERROR)\
printf("GL Error: %s\n", gluErrorString(err)); }

DepthViewer::DepthViewer(QWidget *parent)
: QGLWidget(parent)
{
	m_texture_id = 0;
	m_pbo_id = 0;
	m_pbo_buffer.data = nullptr;
}

DepthViewer::~DepthViewer()
{

}

void DepthViewer::setImage_h(const dfusion::depthtype* image_h, int w, int h)
{
	makeCurrent();

	for (size_t i_pixel = 0; i_pixel < m_depthColors_h.size(); i_pixel++)
	{
		float val = (float)(image_h[i_pixel] - 300) / 700;
		ldp::Float3 c = calcTemperatureJet(val);
		m_depthColors_h[i_pixel] = ldp::UChar4(c[0] * 255, c[1] * 255, c[2] * 255, 255);
	}

	m_gl_func.glBindTexture(GL_TEXTURE_2D, m_texture_id);
	m_gl_func.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_depthColors_h.data());
	updateGL();
}

void DepthViewer::setImage_d(PtrStepSz<dfusion::depthtype> image_d)
{
	makeCurrent();
	cudaSafeCall(cudaGLMapBufferObject((void**)&m_pbo_buffer.data, m_pbo_id));
	dfusion::calc_temperature_jet(image_d, m_pbo_buffer, 300, 700);	cudaSafeCall(cudaGLUnmapBufferObject(m_pbo_id));

	m_gl_func.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo_id);
	m_gl_func.glBindTexture(GL_TEXTURE_2D, m_texture_id);
	m_gl_func.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_pbo_buffer.cols,
		m_pbo_buffer.rows, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	CHECK_GL_ERROR;
	updateGL();
}

void DepthViewer::initializeGL()
{
	makeCurrent();
	m_gl_func.initializeOpenGLFunctions();
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glClearColor(0.f, 0.f, 0.f, 1.0f);
	m_camera.setFrustum(0, 1, 0, 1, 0.1, 100);
	m_camera.enableOrtho(true);
	m_camera.setFrustum(0, 1, 0, 1, 0.1, 100);
	m_camera.lookAt(ldp::Float3(0, 0, 1), ldp::Float3(0, 0, 0), ldp::Float3(0, 1, 0));

	// gen texture
	m_gl_func.glBindTexture(GL_TEXTURE_2D, 0);
	m_gl_func.glGenTextures(1, &m_texture_id);
	m_gl_func.glBindTexture(GL_TEXTURE_2D, m_texture_id);
	m_gl_func.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl_func.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl_func.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, dfusion::KINECT_WIDTH, dfusion::KINECT_HEIGHT,
		0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	m_gl_func.glBindTexture(GL_TEXTURE_2D, 0);
	m_depthColors_h.resize(dfusion::KINECT_WIDTH*dfusion::KINECT_HEIGHT);

	// gen buffer and map to cuda
	m_gl_func.glGenBuffers(1, &m_pbo_id);
	m_gl_func.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo_id);	m_gl_func.glBufferData(GL_PIXEL_UNPACK_BUFFER,
		dfusion::KINECT_WIDTH * dfusion::KINECT_HEIGHT * 4,
		NULL, GL_DYNAMIC_COPY);
	cudaSafeCall(cudaGLRegisterBufferObject(m_pbo_id));
	m_pbo_buffer.rows = dfusion::KINECT_HEIGHT;
	m_pbo_buffer.cols = dfusion::KINECT_WIDTH;
	m_pbo_buffer.step = dfusion::KINECT_WIDTH * sizeof(uchar4);
	m_gl_func.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

}

void DepthViewer::resizeGL(int w, int h)
{
	float aspect = w / (float)(h ? h : 1);
	float scale = std::min(float(w) / float(640), float(h) / float(480));
	float w1 = float(640 * scale);
	float h1 = float(480 * scale);
	m_camera.setViewPort((w - w1) / 2, (w - w1) / 2 + w1, (h - h1) / 2, (h - h1) / 2 + h1);
}

void DepthViewer::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	m_camera.apply();

	glBegin(GL_QUADS);
	glTexCoord2d(0, 0);
	glVertex2d(0, 0);
	glTexCoord2d(1, 0);
	glVertex2d(1, 0);
	glTexCoord2d(1, 1);
	glVertex2d(1, 1);
	glTexCoord2d(0, 1);
	glVertex2d(0, 1);
	glEnd();

}
