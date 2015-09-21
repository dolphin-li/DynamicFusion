#include "glew.h"
#include "glut.h"
#include "GpuMesh.h"
#include <cuda_gl_interop.h>
#include <cudagl.h>
#include "ObjMesh.h"
#include "Camera.h"
#include "CFreeImage.h"
#include "glsl\glsl.h"
#include "glsl\glslprogram.h"

#define CHECK_GL_ERROR(str) {\
	GLenum err = glGetError(); \
if (err != GL_NO_ERROR)\
	printf("[%s]GL Error: %d=%s\n", str, err, gluErrorString(err)); }
#define CHECK_GLEW_ERROR(err, str) {\
if (err != GLEW_NO_ERROR)\
	printf("[%s]Glew Error: %s\n", str, glewGetErrorString(err)); }
#define CHECK_NOT_EQUAL(a, b) {\
if (a == b){\
printf("CHECK FAILED: %s == %s\n", #a, #b); throw std::exception();}}

namespace dfusion
{
//#define ENABLE_SHOW_DEBUG
	HDC g_hdc;
	HGLRC g_glrc;
	cwc::glShader* g_shader_depth;
	cwc::glShaderObject* g_depth_vshader;
	cwc::glShaderObject* g_depth_fshader;
	
#pragma region --shaders
	// for depth buffer rendering
	const static char* g_vshader_depth_src =
		"varying vec4 pos;\n\
		void main()\n\
		{\n\
			gl_Position = gl_ModelViewProjectionMatrix  * gl_Vertex;\n\
			pos = gl_Position;\n\
		}\n";
	const static char* g_fshader_depth_src =
		"varying vec4 pos;\n\
		void main()\n\
		{\n\
			float depth = (pos.z / pos.w + 1.0) * 0.5;\n\
			gl_FragColor.r = float(int(depth*65525.0)&0xff)/255.0;\n\
			gl_FragColor.g = float((int(depth*65525.0)&0xff00)>>8)/255.0;\n\
			gl_FragColor.b = depth*65525.0 - float(int(depth*65525.0));\n\
			gl_FragColor.a = 0.0;\n\
		}\n";																											
#pragma endregion

#pragma region --create gl context
	HWND g_hwnd;
	HPALETTE g_hpalette;
	GpuMesh* g_testmesh;
	Camera g_testCam;

	void setupPixelFormat(HDC hDC)
	{
		PIXELFORMATDESCRIPTOR pfd = {
			sizeof(PIXELFORMATDESCRIPTOR),  /* size */
			1,                              /* version */
			PFD_SUPPORT_OPENGL |
			PFD_DRAW_TO_WINDOW |
			PFD_DOUBLEBUFFER,               /* support double-buffering */
			PFD_TYPE_RGBA,                  /* color type */
			16,                             /* prefered color depth */
			0, 0, 0, 0, 0, 0,               /* color bits (ignored) */
			0,                              /* no alpha buffer */
			0,                              /* alpha bits (ignored) */
			0,                              /* no accumulation buffer */
			0, 0, 0, 0,                     /* accum bits (ignored) */
			16,                             /* depth buffer */
			0,                              /* no stencil buffer */
			0,                              /* no auxiliary buffers */
			PFD_MAIN_PLANE,                 /* main layer */
			0,                              /* reserved */
			0, 0, 0,                        /* no layer, visible, damage masks */
		};
		int pixelFormat;

		pixelFormat = ChoosePixelFormat(hDC, &pfd);
		if (pixelFormat == 0) {
			MessageBox(WindowFromDC(hDC), L"ChoosePixelFormat failed.", L"Error",
				MB_ICONERROR | MB_OK);
			exit(1);
		}

		if (SetPixelFormat(hDC, pixelFormat, &pfd) != TRUE) {
			MessageBox(WindowFromDC(hDC), L"SetPixelFormat failed.", L"Error",
				MB_ICONERROR | MB_OK);
			exit(1);
		}
	}

	void setupPalette(HDC hDC)
	{
		int pixelFormat = GetPixelFormat(hDC);
		PIXELFORMATDESCRIPTOR pfd;
		LOGPALETTE* pPal;
		int paletteSize;

		DescribePixelFormat(hDC, pixelFormat, sizeof(PIXELFORMATDESCRIPTOR), &pfd);

		if (pfd.dwFlags & PFD_NEED_PALETTE) {
			paletteSize = 1 << pfd.cColorBits;
		}
		else {
			return;
		}

		pPal = (LOGPALETTE*)
			malloc(sizeof(LOGPALETTE)+paletteSize * sizeof(PALETTEENTRY));
		pPal->palVersion = 0x300;
		pPal->palNumEntries = paletteSize;

		/* build a simple RGB color palette */
		{
			int redMask = (1 << pfd.cRedBits) - 1;
			int greenMask = (1 << pfd.cGreenBits) - 1;
			int blueMask = (1 << pfd.cBlueBits) - 1;
			int i;

			for (i = 0; i < paletteSize; ++i) {
				pPal->palPalEntry[i].peRed =
					(((i >> pfd.cRedShift) & redMask) * 255) / redMask;
				pPal->palPalEntry[i].peGreen =
					(((i >> pfd.cGreenShift) & greenMask) * 255) / greenMask;
				pPal->palPalEntry[i].peBlue =
					(((i >> pfd.cBlueShift) & blueMask) * 255) / blueMask;
				pPal->palPalEntry[i].peFlags = 0;
			}
		}

		g_hpalette = CreatePalette(pPal);
		free(pPal);

		if (g_hpalette) {
			SelectPalette(hDC, g_hpalette, FALSE);
			RealizePalette(hDC);
		}
	}

	LRESULT APIENTRY WindowProc(
		HWND hWnd,
		UINT message,
		WPARAM wParam,
		LPARAM lParam)
	{
		switch (message)
		{
		case WM_CREATE:
			/* initialize OpenGL rendering */
			g_hdc = GetDC(hWnd);
			setupPixelFormat(g_hdc);
			setupPalette(g_hdc);
			g_glrc = wglCreateContext(g_hdc);
			wglMakeCurrent(g_hdc, g_glrc);
			CHECK_GLEW_ERROR(glewInit(), "GpuMesh");
			return 0;
		case WM_DESTROY:
			/* finish OpenGL rendering */
			if (g_glrc)
			{
				wglMakeCurrent(NULL, NULL);
				wglDeleteContext(g_glrc);
			}
			if (g_hpalette)
			{
				DeleteObject(g_hpalette);
			}
			ReleaseDC(hWnd, g_hdc);
			PostQuitMessage(0);
			return 0;
		case WM_PALETTECHANGED:
			/* realize palette if this is *not* the current window */
			if (g_glrc && g_hpalette && (HWND)wParam != hWnd) {
				UnrealizeObject(g_hpalette);
				SelectPalette(g_hdc, g_hpalette, FALSE);
				RealizePalette(g_hdc);
				if (g_testmesh)
					g_testmesh->renderToImg(g_testCam, LightSource(), ColorMap());
				break;
			}
			break;
		case WM_QUERYNEWPALETTE:
			/* realize palette if this is the current window */
			if (g_glrc && g_hpalette) {
				UnrealizeObject(g_hpalette);
				SelectPalette(g_hdc, g_hpalette, FALSE);
				RealizePalette(g_hdc);
				if (g_testmesh)
					g_testmesh->renderToImg(g_testCam, LightSource(), ColorMap());
				return TRUE;
			}
			break;
		case WM_PAINT:
			if (1)
			{
				PAINTSTRUCT ps;
				BeginPaint(hWnd, &ps);
				if (g_glrc && g_testmesh)
					g_testmesh->renderToImg(g_testCam, LightSource(), ColorMap());
				EndPaint(hWnd, &ps);
				return 0;
			}
			break;
		default:
			break;
		}
		return DefWindowProc(hWnd, message, wParam, lParam);
	}

	static void create_context_gpumesh()
	{
		LPCWSTR className = L"GpuMesh Hidden Window";
		WNDCLASS wc = {};
		wc.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
		wc.lpfnWndProc = WindowProc;
		wc.cbClsExtra = 0;
		wc.cbWndExtra = 0;
		wc.hInstance = GetModuleHandle(NULL);
		wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
		wc.hCursor = LoadCursor(NULL, IDC_ARROW);
		wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
		wc.lpszMenuName = NULL;
		wc.lpszClassName = className;
		RegisterClass(&wc);
			
		// Create the window.
		g_hwnd = CreateWindowEx(
			0,                              // Optional window styles.
			className,                     // Window class
			L"GpuMesh Hidden Window",    // Window text
			WS_OVERLAPPEDWINDOW,            // Window style
			// Size and position
			CW_USEDEFAULT, CW_USEDEFAULT, KINECT_WIDTH, KINECT_HEIGHT,
			NULL,       // Parent window    
			NULL,       // Menu
			::GetModuleHandle(NULL),  // Instance handle
			NULL        // Additional application data
			);
#ifdef ENABLE_SHOW_DEBUG
		g_testCam.setViewPort(0, KINECT_WIDTH, 0, KINECT_HEIGHT);
		float aspect = KINECT_WIDTH / (float)(KINECT_WIDTH);
		g_testCam.setPerspective(45.6f, aspect, 0.3f, 30.f);
		g_testCam.lookAt(ldp::Float3(1,2,2), ldp::Float3(0,0,0), ldp::Float3(0,1,0));
#endif

		if (g_glrc == nullptr)
			throw std::exception("GpuMesh: create GLRC failed.");

		// create shader
		g_depth_fshader = new cwc::aFragmentShader();
		g_depth_vshader = new cwc::aVertexShader();
		g_shader_depth = new cwc::glShader();
		g_depth_fshader->loadFromMemory(g_fshader_depth_src);
		g_depth_vshader->loadFromMemory(g_vshader_depth_src);
		g_depth_fshader->compile();
		g_depth_vshader->compile();
		g_shader_depth->addShader(g_depth_vshader);
		g_shader_depth->addShader(g_depth_fshader);
		g_shader_depth->link();
		printf("%s\n", g_shader_depth->getLinkerLog());
	}
#pragma endregion

	GpuMesh::GpuMesh()
	{
		m_verts_d = nullptr;
		m_normals_d = nullptr;
		m_cuda_res = nullptr;
		m_vbo_id = 0;
		m_num = 0;
		m_width = 0;
		m_height = 0;

		m_render_fbo_id = 0;
		m_render_texture_id = 0;
		m_render_depth_id = 0;
		m_render_fbo_pbo_id = 0;
		m_cuda_res_fbo = nullptr;
	}

	GpuMesh::GpuMesh(GpuMesh& rhs)
	{
		copyFrom(rhs);
	}

	GpuMesh::~GpuMesh()
	{
		release();
		releaseRenderer();
	}

	void GpuMesh::create(size_t n)
	{
		if (m_num != n)
			release();

		if (n == 0)
			return;

		if (m_vbo_id == 0)
		{
			if (g_hdc == nullptr)
				create_context_gpumesh();

			if (!wglMakeCurrent(g_hdc, g_glrc))
				throw std::exception("wglMakeCurrent error");

			// create buffer object
			do{
				glGenBuffers(1, &m_vbo_id);
				CHECK_NOT_EQUAL(m_vbo_id, 0);
			} while (is_cuda_pbo_vbo_id_used_push_new(m_vbo_id));
			glBindBuffer(GL_ARRAY_BUFFER, m_vbo_id);
			glBufferData(GL_ARRAY_BUFFER, 2 * n*sizeof(PointType), 0, GL_DYNAMIC_DRAW);
			if (n)// when n==0, it may crash.
			{
				cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_cuda_res, m_vbo_id,
					cudaGraphicsMapFlagsNone));
			}
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			m_num = n;
		}
	}

	void GpuMesh::release()
	{
		if (m_vbo_id != 0)
			glDeleteBuffers(1, &m_vbo_id);
		if (m_cuda_res)
			cudaSafeCall(cudaGraphicsUnregisterResource(m_cuda_res));
		m_verts_d = nullptr;
		m_normals_d = nullptr;
		m_cuda_res = nullptr;
		m_vbo_id = 0;
		m_num = 0;
	}

	void GpuMesh::copyFrom(GpuMesh& rhs)
	{
		create(rhs.num());

		rhs.lockVertsNormals();
		lockVertsNormals();

		cudaMemcpy(verts(), rhs.verts(), num() * 2 * sizeof(PointType), cudaMemcpyDeviceToDevice);

		unlockVertsNormals();
		rhs.unlockVertsNormals();
	}

	void GpuMesh::lockVertsNormals()
	{
		// has been locked before
		if (m_verts_d)
			return;

		// no available data
		if (m_cuda_res == nullptr)
			return;

		size_t num_bytes = 0;
		cudaSafeCall(cudaGraphicsMapResources(1, &m_cuda_res, 0));
		cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&m_verts_d, &num_bytes, m_cuda_res));
		m_normals_d = m_verts_d + m_num;
	}
	void GpuMesh::unlockVertsNormals()
	{
		if (m_verts_d == nullptr)
			return;
		cudaSafeCall(cudaGraphicsUnmapResources(1, &m_cuda_res, 0));
		m_verts_d = nullptr;
		m_normals_d = nullptr;
	}

	void GpuMesh::toObjMesh(ObjMesh& omesh)
	{
		omesh.clear();

		lockVertsNormals();

		omesh.vertex_list.resize(num());
		omesh.vertex_normal_list.resize(num());
		std::vector<PointType> tmpvert, tmpnorm;
		tmpvert.resize(num());
		tmpnorm.resize(num());
		cudaMemcpy(tmpvert.data(), verts(), num()*sizeof(PointType), cudaMemcpyDeviceToHost);
		cudaMemcpy(tmpnorm.data(), normals(), num()*sizeof(PointType), cudaMemcpyDeviceToHost);

		unlockVertsNormals();

		for (size_t i = 0; i < num(); i++)
		{
			omesh.vertex_list[i] = ldp::Float3(tmpvert[i].x, tmpvert[i].y, tmpvert[i].z);
			omesh.vertex_normal_list[i] = ldp::Float3(tmpnorm[i].x, tmpnorm[i].y, tmpnorm[i].z);
		}

		omesh.face_list.resize(omesh.vertex_list.size() / 3);
		for (size_t fid = 0; fid < omesh.face_list.size(); fid++)
		{
			ObjMesh::obj_face &f = omesh.face_list[fid];
			f.material_index = -1;
			f.vertex_count = 3;
			f.vertex_index[0] = fid * 3;
			f.vertex_index[1] = fid * 3 + 1;
			f.vertex_index[2] = fid * 3 + 2;
		}
	}

	void GpuMesh::createRenderer(int w, int h)
	{
		if (w != m_width || h != m_height)
			releaseRenderer();

		if (m_render_fbo_id == 0)
		{
			if (g_hdc == nullptr)
			{
				create_context_gpumesh();
				if (!wglMakeCurrent(g_hdc, g_glrc))
					throw std::exception("wglMakeCurrent error");
			}

			m_width = w;
			m_height = h;

			// create render object
			glGenFramebuffersEXT(1, &m_render_fbo_id);
			CHECK_NOT_EQUAL(m_render_fbo_id, 0);
			glGenTexturesEXT(1, &m_render_texture_id);
			CHECK_NOT_EQUAL(m_render_texture_id, 0);
			glGenRenderbuffersEXT(1, &m_render_depth_id);
			CHECK_NOT_EQUAL(m_render_depth_id, 0);

			glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_render_fbo_id);

			// the render texture
			glBindTextureEXT(GL_TEXTURE_2D, m_render_texture_id);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h,
				0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
				GL_TEXTURE_2D, m_render_texture_id, 0);
			glBindTextureEXT(GL_TEXTURE_2D, 0);

			// The depth buffer
			glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_render_depth_id);
			glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, w, h);
			glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
				GL_RENDERBUFFER_EXT, m_render_depth_id);
			glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
			glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

			// Always check that our framebuffer is ok
			if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT) != GL_FRAMEBUFFER_COMPLETE_EXT)
				throw std::exception("frameBuffer creating failed");

			// create pbo cuda
			glGenBuffers(1, &m_render_fbo_pbo_id);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_render_fbo_pbo_id);			glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * sizeof(PointType)* 2,
				NULL, GL_DYNAMIC_COPY);
			cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_cuda_res_fbo, m_render_fbo_pbo_id,
				cudaGraphicsRegisterFlagsReadOnly));
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

#ifdef ENABLE_SHOW_DEBUG
			g_testmesh = this;
			ShowWindow(g_hwnd, SHOW_OPENWINDOW);
#endif
		}
	}

	void GpuMesh::releaseRenderer()
	{
		if (m_render_fbo_id != 0)
		{
			glDeleteTextures(1, &m_render_texture_id);
			glDeleteRenderbuffers(1, &m_render_depth_id);
			glDeleteFramebuffers(1, &m_render_fbo_id);
			glDeleteBuffers(1, &m_render_fbo_pbo_id);
			cudaSafeCall(cudaGraphicsUnregisterResource(m_cuda_res_fbo));
		}
		m_width = 0;
		m_height = 0;
		m_render_fbo_id = 0;
		m_render_texture_id = 0;
		m_render_depth_id = 0;
		m_render_fbo_pbo_id = 0;
		m_cuda_res_fbo = nullptr;
	}

	void GpuMesh::renderToImg(const Camera& camera, LightSource light, ColorMap& img)
	{
		if (!wglMakeCurrent(g_hdc, g_glrc))
			throw std::exception("wglMakeCurrent error");

		createRenderer(std::lroundf(abs(camera.getViewPortRight()-camera.getViewPortLeft())),
			std::lroundf(abs(camera.getViewPortBottom() - camera.getViewPortTop())));

#ifndef ENABLE_SHOW_DEBUG
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_render_fbo_id);
#endif
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_DEPTH_TEST);

		ldp::Float3 sv = ldp::Float3(light.diffuse.x,
			light.diffuse.y, light.diffuse.z)*camera.getScalar();
		glLightfv(GL_LIGHT0, GL_DIFFUSE, sv.ptr());
		ldp::Float3 sa = 0.f;
		glLightfv(GL_LIGHT0, GL_AMBIENT, &light.amb.x);
		glLightfv(GL_LIGHT0, GL_SPECULAR, &light.spec.x);

		camera.apply();

		unlockVertsNormals();

		glBindBuffer(GL_ARRAY_BUFFER, m_vbo_id);
		glVertexPointer(3, GL_FLOAT, sizeof(PointType), 0);
		glEnableClientState(GL_VERTEX_ARRAY);

		size_t shift = m_num*sizeof(PointType);
		glNormalPointer(GL_FLOAT, sizeof(PointType), (GLvoid*)shift);
		glEnableClientState(GL_NORMAL_ARRAY);

		glColor3f(1.0, 1.0, 1.0);
		glDrawArrays(GL_TRIANGLES, 0, m_num);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glPopAttrib();
		SwapBuffers(g_hdc);

#ifndef ENABLE_SHOW_DEBUG
		glBindBuffer(GL_PIXEL_PACK_BUFFER, m_render_fbo_pbo_id);
		glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

		uchar4* gldata = nullptr;
		size_t num_bytes = 0;
		cudaSafeCall(cudaGraphicsMapResources(1, &m_cuda_res_fbo, 0));
		cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gldata, &num_bytes, m_cuda_res_fbo));
		copy_invert_y(gldata, img);
		cudaSafeCall(cudaGraphicsUnmapResources(1, &m_cuda_res_fbo, 0));

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
#endif
		CHECK_GL_ERROR("renderToImg");
	}

	void GpuMesh::renderToDepth(const Camera& camera, DepthMap& img)
	{
		if (!wglMakeCurrent(g_hdc, g_glrc))
			throw std::exception("wglMakeCurrent error");

		createRenderer(std::lroundf(abs(camera.getViewPortRight() - camera.getViewPortLeft())),
			std::lroundf(abs(camera.getViewPortBottom() - camera.getViewPortTop())));

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_render_fbo_id);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);

		camera.apply();

		unlockVertsNormals();

		g_shader_depth->begin();

		glBindBuffer(GL_ARRAY_BUFFER, m_vbo_id);
		glVertexPointer(3, GL_FLOAT, sizeof(PointType), 0);
		glEnableClientState(GL_VERTEX_ARRAY);

		size_t shift = m_num*sizeof(PointType);
		glNormalPointer(GL_FLOAT, sizeof(PointType), (GLvoid*)shift);
		glEnableClientState(GL_NORMAL_ARRAY);

		glColor3f(1.0, 1.0, 1.0);
		glDrawArrays(GL_TRIANGLES, 0, m_num);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glPopAttrib();
		SwapBuffers(g_hdc);

		g_shader_depth->end();

		glBindBuffer(GL_PIXEL_PACK_BUFFER, m_render_fbo_pbo_id);
		glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

		uchar4* gldata = nullptr;
		size_t num_bytes = 0;
		cudaSafeCall(cudaGraphicsMapResources(1, &m_cuda_res_fbo, 0));
		cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gldata, &num_bytes, m_cuda_res_fbo));
		const float s1 = 2.f*camera.getFrustumNear()*camera.getFrustumFar() /
			(camera.getFrustumNear() - camera.getFrustumFar());
		const float s2 = (camera.getFrustumNear() + camera.getFrustumFar()) /
			(camera.getFrustumNear() - camera.getFrustumFar());
		copy_gldepth_to_depthmap(gldata, img, s1, s2, camera.getFrustumNear());
		cudaSafeCall(cudaGraphicsUnmapResources(1, &m_cuda_res_fbo, 0));


		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
		CHECK_GL_ERROR("renderToDepth");
	}
}