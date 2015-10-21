#include "glew.h"
#include "DepthViewer.h"
#include "util.h"
#include <cuda_gl_interop.h>
#include <cudagl.h>

#define CHECK_GL_ERROR(str) {\
GLenum err = glGetError();\
if (err != GL_NO_ERROR)\
printf("[%s]GL Error: %d=%s\n", str, err, gluErrorString(err)); }
#define CHECK_NOT_EQUAL(a, b) {\
if (a == b){\
printf("CHECK FAILED: %s == %s\n", #a, #b); throw std::exception();}}

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

	glBindTexture(GL_TEXTURE_2D, m_texture_id);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_depthColors_h.data());
	updateGL();
}

void DepthViewer::setImage_d(PtrStepSz<dfusion::depthtype> image_d)
{
	makeCurrent();

	size_t num_bytes = 0;
	cudaSafeCall(cudaGraphicsMapResources(1, &m_pbo_cuda_res, 0));
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&m_pbo_buffer.data, &num_bytes, m_pbo_cuda_res));
	dfusion::calc_temperature_jet(image_d, m_pbo_buffer, 300, 700);
	cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pbo_cuda_res, 0));

	m_gl_func->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo_id);
	glBindTexture(GL_TEXTURE_2D, m_texture_id);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_pbo_buffer.cols,
		m_pbo_buffer.rows, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	m_gl_func->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	CHECK_GL_ERROR("setImage_d");
	updateGL();
}

void DepthViewer::setNormal_d(const dfusion::MapArr& image_d)
{
	makeCurrent();

	size_t num_bytes = 0;
	cudaSafeCall(cudaGraphicsMapResources(1, &m_pbo_cuda_res, 0));
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&m_pbo_buffer.data, &num_bytes, m_pbo_cuda_res));
	dfusion::ColorMap map(dfusion::KINECT_HEIGHT, dfusion::KINECT_WIDTH, 
		m_pbo_buffer, dfusion::KINECT_WIDTH*sizeof(uchar4));
	dfusion::generateNormalMap(image_d, map);
	cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pbo_cuda_res, 0));

	m_gl_func->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo_id);
	glBindTexture(GL_TEXTURE_2D, m_texture_id);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_pbo_buffer.cols,
		m_pbo_buffer.rows, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	m_gl_func->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	CHECK_GL_ERROR("setImage_d");
	updateGL();
}

void DepthViewer::initializeGL()
{
	makeCurrent();
	m_gl_func = new QGLFunctions(this->context());
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glClearColor(0.f, 0.f, 0.f, 1.0f);
	m_camera.setFrustum(0, 1, 0, 1, 0.1, 100);
	m_camera.enableOrtho(true);
	m_camera.setFrustum(0, 1, 0, 1, 0.1, 100);
	m_camera.lookAt(ldp::Float3(0, 0, 1), ldp::Float3(0, 0, 0), ldp::Float3(0, 1, 0));

	// gen texture
	glBindTexture(GL_TEXTURE_2D, 0);
	glGenTextures(1, &m_texture_id);
	glBindTexture(GL_TEXTURE_2D, m_texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, dfusion::KINECT_WIDTH, dfusion::KINECT_HEIGHT,
		0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	m_depthColors_h.resize(dfusion::KINECT_WIDTH*dfusion::KINECT_HEIGHT);

	// gen buffer and map to cuda
	do{
		m_gl_func->glGenBuffers(1, &m_pbo_id);
		CHECK_NOT_EQUAL(m_pbo_id, 0);
	} while (dfusion::is_cuda_pbo_vbo_id_used_push_new(m_pbo_id));

	m_gl_func->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo_id);	m_gl_func->glBufferData(GL_PIXEL_UNPACK_BUFFER,
		dfusion::KINECT_WIDTH * dfusion::KINECT_HEIGHT * 4,
		NULL, GL_DYNAMIC_COPY);
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_pbo_cuda_res, m_pbo_id,
		cudaGraphicsMapFlagsWriteDiscard));
	m_pbo_buffer.rows = dfusion::KINECT_HEIGHT;
	m_pbo_buffer.cols = dfusion::KINECT_WIDTH;
	m_pbo_buffer.step = dfusion::KINECT_WIDTH * sizeof(uchar4);
	m_gl_func->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

}

void DepthViewer::resizeGL(int w, int h)
{
	float aspect = w / (float)(h ? h : 1);
	float scale = std::min(float(w) / float(dfusion::KINECT_WIDTH), float(h) / float(dfusion::KINECT_HEIGHT));
	float w1 = float(dfusion::KINECT_WIDTH * scale);
	float h1 = float(dfusion::KINECT_HEIGHT * scale);
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
