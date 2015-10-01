#include "RayCastViewer.h"
#include "global_data_holder.h"
#include <cuda_gl_interop.h>
#include <cudagl.h>
#include "glut.h"
#include "MicrosoftKinect.h"
#include "kinect_util.h"

#define CHECK_GL_ERROR(str) {\
	GLenum err = glGetError(); \
if (err != GL_NO_ERROR)\
	printf("[%s], GL Error: %s\n", str, gluErrorString(err)); }
#define CHECK_NOT_EQUAL(a, b) {\
if (a == b){\
	printf("CHECK FAILED: %s == %s\n", #a, #b); throw std::exception();}}

RayCastViewer::RayCastViewer(QWidget *parent)
: QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
	m_rootTrans.eye();
	m_defaultRootTrans.eye();
	m_dataScale = 1;
	m_id = 0;

	m_defaultCameraLocation = ldp::Float3(0, 0, 0);
	m_defaultCameraDirection = ldp::Float3(0, 0, -1);
	m_defaultCameraUp = ldp::Float3(0, 1, 0);

	setMouseTracking(true);

	m_texture_id = 0;
	m_pbo_id = 0;
	m_pbo_buffer.data = nullptr;
}

RayCastViewer::~RayCastViewer()
{
}

void RayCastViewer::initializeGL()
{
	makeCurrent();
	m_gl_func = new QGLFunctions(this->context());
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	ldp::Float3 sv = ldp::Float3(g_dataholder.m_lights.diffuse.x,
		g_dataholder.m_lights.diffuse.y,
		g_dataholder.m_lights.diffuse.z);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, sv.ptr());
	glLightfv(GL_LIGHT0, GL_AMBIENT, &g_dataholder.m_lights.amb.x);
	glLightfv(GL_LIGHT0, GL_SPECULAR, &g_dataholder.m_lights.spec.x);

	m_camera.lookAt(m_defaultCameraLocation, m_defaultCameraLocation + 
		m_defaultCameraDirection, m_defaultCameraUp);
	m_camera.setPerspective(KINECT_DEPTH_V_FOV, 1, KINECT_NEAREST_METER, 30.f);

	// generate texture
	glBindTexture(GL_TEXTURE_2D, 0);
	glGenTextures(1, &m_texture_id);
	glBindTexture(GL_TEXTURE_2D, m_texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, dfusion::KINECT_WIDTH, dfusion::KINECT_HEIGHT,
		0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

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

void RayCastViewer::resizeGL(int w, int h)
{
	float aspect = w / (float)(h ? h : 1);
	float scale = std::min(float(w) / float(m_pbo_buffer.cols), float(h) / float(m_pbo_buffer.rows));
	float w1 = float(m_pbo_buffer.cols * scale);
	float h1 = float(m_pbo_buffer.rows * scale);
	m_camera.setViewPort((w - w1) / 2, (w - w1) / 2 + w1, (h - h1) / 2, (h - h1) / 2 + h1);
	m_camera.setPerspective(m_camera.getFov(), aspect, KINECT_NEAREST_METER, 30.f);
}

void RayCastViewer::setSameView(const RayCastViewer* other)
{
	m_camera = other->m_camera;
	m_rootTrans = other->m_rootTrans;
	updateGL();
}

void RayCastViewer::paintGL()
{
	m_camera.apply();
	glMultMatrixf(m_rootTrans.ptr());


	glClearColor(0.f, 0.f, 0.f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	renderRayCasting();


#if 1
	// render the axis
	{
		Camera axisCam = m_camera;
		axisCam.setScalar(1);
		axisCam.setViewPort(0, width() / 4, height()*3 / 4, height());
		axisCam.setLocation(axisCam.getLocation().normalize()*1.2);
		axisCam.apply();
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		glEnable(GL_LINE_SMOOTH);
		glLineWidth(4);
		glBegin(GL_LINES);
		glColor3f(1, 0, 0);
		glVertex3f(0, 0, 0);
		glVertex3f(1, 0, 0);
		glColor3f(0, 1, 0);
		glVertex3f(0, 0, 0);
		glVertex3f(0, 1, 0);
		glColor3f(0, 0, 1);
		glVertex3f(0, 0, 0);
		glVertex3f(0, 0, 1);
		glEnd();
		glPopAttrib();
	}
#endif

}

void RayCastViewer::getCameraInfo(Camera& cam)const
{
	cam = m_camera;
	cam.setModelViewMatrix(cam.getModelViewMatrix()*m_rootTrans);
}

void RayCastViewer::setCameraInfo(const Camera& cam)
{
	m_camera = cam;
	m_camera.setModelViewMatrix(cam.getModelViewMatrix()*m_rootTrans.inv());
	updateGL();
}

void RayCastViewer::setRayCastingShadingImage(const dfusion::ColorMap& img)
{
	makeCurrent();

	// dynamic resize
	if (img.cols() != m_pbo_buffer.cols || img.rows() != m_pbo_buffer.rows)
	{
		// resize pbo
		m_gl_func->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo_id);		m_gl_func->glBufferData(GL_PIXEL_UNPACK_BUFFER,
			img.cols() * img.rows() * 4,
			NULL, GL_DYNAMIC_COPY);
		cudaSafeCall(cudaGraphicsUnregisterResource(m_pbo_cuda_res));
		cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_pbo_cuda_res, m_pbo_id,
			cudaGraphicsMapFlagsWriteDiscard));
		m_pbo_buffer.rows = img.rows();
		m_pbo_buffer.cols = img.cols();
		m_pbo_buffer.step = img.cols() * sizeof(uchar4);
		m_gl_func->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// resize texture
		glBindTexture(GL_TEXTURE_2D, m_texture_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img.cols(), img.rows(),
			0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
		glBindTexture(GL_TEXTURE_2D, 0);

		resizeGL(width(), height());
	}

	size_t num_bytes = 0;
	cudaSafeCall(cudaGraphicsMapResources(1, &m_pbo_cuda_res, 0));
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&m_pbo_buffer.data, &num_bytes, m_pbo_cuda_res));	dfusion::copyColorMapToPbo(img, m_pbo_buffer);
	cudaSafeCall(cudaGraphicsUnmapResources(1, &m_pbo_cuda_res, 0));

	m_gl_func->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo_id);
	glBindTexture(GL_TEXTURE_2D, m_texture_id);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_pbo_buffer.cols,
		m_pbo_buffer.rows, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	m_gl_func->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	CHECK_GL_ERROR("setRayCastingShadingImage");
	updateGL();
}

void RayCastViewer::renderRayCasting()
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	Camera cam = m_camera;
	cam.enableOrtho(true);
	cam.setFrustum(0, 1, 0, 1, -1, 1);
	cam.setModelViewMatrix(ldp::Mat4f().eye());
	cam.apply();

	glEnable(GL_TEXTURE_2D);
	glColor3f(1, 1, 1);
	glDisable(GL_LIGHTING);
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

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();
}

void RayCastViewer::mousePressEvent(QMouseEvent *ev)
{
	setFocus();
	m_lastPos = ev->pos();
	m_buttons = ev->buttons();
	m_lastMousePressPos = ev->pos();

	if (ev->button() == Qt::MouseButton::LeftButton)
	{
		// mesh roate begin
		m_camera.arcballClick(ldp::Float2(ev->pos().x(), ev->pos().y()));
	}

	// move operation
	if (ev->modifiers() == Qt::NoModifier)
	{
		if (ev->button() == Qt::MouseButton::MiddleButton)
		{
			m_camera.lookAt(m_defaultCameraLocation, m_defaultCameraLocation + m_defaultCameraDirection, m_defaultCameraUp);
			m_camera.setScalar(1);
			m_rootTrans = m_defaultRootTrans;
			m_camera.setPerspective(KINECT_DEPTH_V_FOV, m_camera.getAspect(),
				m_camera.getFrustumNear(), m_camera.getFrustumFar());
		}
	}
}

void RayCastViewer::keyPressEvent(QKeyEvent*ev)
{
	bool noMod = ((ev->modifiers() & Qt::SHIFT) == 0)
		&& ((ev->modifiers() & Qt::CTRL) == 0)
		& ((ev->modifiers() & Qt::ALT) == 0);
	switch (ev->key())
	{
	default:
		break;
	}
}

void RayCastViewer::mouseReleaseEvent(QMouseEvent *ev)
{
	// clear buttons
	m_buttons = Qt::NoButton;

	// backup last position
	m_lastPos = ev->pos();
}

void RayCastViewer::mouseMoveEvent(QMouseEvent*ev)
{
	// move operation
	if (ev->modifiers() == Qt::NoModifier)
	{
		if (m_buttons == Qt::MouseButton::LeftButton)
		{
			m_camera.arcballDrag(ldp::Float2(ev->pos().x(), ev->pos().y())); 
		}

		if (m_buttons == Qt::MouseButton::RightButton)
		{
			QPoint dif = ev->pos() - m_lastPos;
			ldp::Float3 t(-(float)dif.x() / width(), (float)dif.y() / height(), 0);
			t = m_camera.getModelViewMatrix().getRotationPart().inv() * t * m_dataScale;
			for (int k = 0; k < 3; k++)
				m_rootTrans(k, 3) -= t[k];
		}
	}


	// backup last position
	m_lastPos = ev->pos();
}

void RayCastViewer::wheelEvent(QWheelEvent*ev)
{
#if 0
	float s = 1.2;
	if (ev->delta() < 0)
		s = 1 / s;

	m_camera.scale(s);
#else
	float s = 1.2;
	if (ev->delta() < 0)
		s = 1 / s;
	m_camera.setPerspective(m_camera.getFov()*s, m_camera.getAspect(), 
		m_camera.getFrustumNear(), m_camera.getFrustumFar());
#endif
}
