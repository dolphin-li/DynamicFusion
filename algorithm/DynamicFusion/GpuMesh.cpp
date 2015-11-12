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
#include "WarpField.h"
#include "TsdfVolume.h"

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
	cwc::glShader* g_shader_node;
	cwc::glShaderObject* g_node_vshader;
	cwc::glShaderObject* g_node_fshader;
	cwc::glShader* g_shader_cano;
	cwc::glShaderObject* g_cano_vshader;
	cwc::glShaderObject* g_cano_fshader;
	
#pragma region --shaders
#define STRCPY(x) #x
	// for depth buffer rendering
	const static char* g_vshader_depth_src = STRCPY(
		varying vec4 pos;
		void main()
		{
			gl_Position = gl_ModelViewProjectionMatrix  * gl_Vertex;
			pos = gl_Position;
		}
	);
	const static char* g_fshader_depth_src = STRCPY(
		varying vec4 pos;
		void main()
		{
			float depth = (pos.z / pos.w + 1.0) * 0.5;
			gl_FragColor.r = depth;
		}
	);	

	// vertex shader for rendering points as shaded spheres
	const char *g_vshader_node_src = STRCPY(
		uniform float pointRadius;  // point size in world space
		uniform float pointScale;   // scale to calculate size in pixels
		void main()
		{
			// calculate window-space point size
			vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
			float dist = length(posEye);
			gl_PointSize = pointRadius * (pointScale / dist);

			gl_TexCoord[0] = gl_MultiTexCoord0;
			gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

			gl_FrontColor = gl_Color;
		}
	);

	// pixel shader for rendering points as shaded spheres
	const char *g_fshader_node_src = STRCPY(
		void main()
		{
			const vec3 lightDir = vec3(0.577, 0.577, 0.577);

			// calculate normal from texture coordinates
			vec3 N;
			N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
			float mag = dot(N.xy, N.xy);

			if (mag > 1.0) discard;   // kill pixels outside circle

			N.z = sqrt(1.0 - mag);

			// calculate lighting
			float diffuse = max(0.0, dot(lightDir, N));

			gl_FragColor = gl_Color *diffuse;
		}
	);

	// vertex shader for rendering cano view
	const char *g_vshader_cano_src = STRCPY(
		varying vec4 color;
		void main()
		{
			gl_Position = gl_ModelViewProjectionMatrix  * gl_Vertex;
			color = gl_Color;
		}
	);

	// pixel shader for rendering cano view
	const char *g_fshader_cano_src = STRCPY(
		varying vec4 color;
		void main()
		{
			gl_FragColor = color;
		}
	);
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
					g_testmesh->renderToImg(g_testCam, LightSource(), ColorMap(), Param());
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
					g_testmesh->renderToImg(g_testCam, LightSource(), ColorMap(), Param());
				return TRUE;
			}
			break;
		case WM_PAINT:
			if (1)
			{
				PAINTSTRUCT ps;
				BeginPaint(hWnd, &ps);
				if (g_glrc && g_testmesh)
					g_testmesh->renderToImg(g_testCam, LightSource(), ColorMap(), Param());
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

		// create depth shader
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
		printf("[depth shader log]: %s\n", g_shader_depth->getLinkerLog());

		// create node shader
		g_node_fshader = new cwc::aFragmentShader();
		g_node_vshader = new cwc::aVertexShader();
		g_shader_node = new cwc::glShader();
		g_node_fshader->loadFromMemory(g_fshader_node_src);
		g_node_vshader->loadFromMemory(g_vshader_node_src);
		g_node_fshader->compile();
		g_node_vshader->compile();
		g_shader_node->addShader(g_node_vshader);
		g_shader_node->addShader(g_node_fshader);
		g_shader_node->link();
		printf("[node shader log]: %s\n", g_shader_node->getLinkerLog());

		// create canonical view shader
		g_cano_fshader = new cwc::aFragmentShader();
		g_cano_vshader = new cwc::aVertexShader();
		g_shader_cano = new cwc::glShader();
		g_cano_fshader->loadFromMemory(g_fshader_cano_src);
		g_cano_vshader->loadFromMemory(g_vshader_cano_src);
		g_cano_fshader->compile();
		g_cano_vshader->compile();
		g_shader_cano->addShader(g_cano_vshader);
		g_shader_cano->addShader(g_cano_fshader);
		g_shader_cano->link();
		printf("[cano shader log]: %s\n", g_shader_cano->getLinkerLog());
	}
#pragma endregion

	GpuMesh::GpuMesh()
	{
		m_verts_d = nullptr;
		m_normals_d = nullptr;
		m_colors_d = nullptr;
		m_cuda_res = nullptr;
		m_vbo_id = 0;
		m_num = 0;
		m_width = 0;
		m_height = 0;
		m_current_buffer_size = 0;
		m_show_color = false;

		m_render_fbo_id = 0;
		m_render_texture_id = 0;
		m_render_depth_id = 0;
		m_render_fbo_pbo_id = 0;
		m_cuda_res_fbo = nullptr;

		m_vbo_id_warpnodes = 0; 
		m_cuda_res_warp = nullptr;
	}

	GpuMesh::GpuMesh(GpuMesh& rhs)
	{
		copyFrom(rhs);
	}

	GpuMesh::~GpuMesh()
	{
		release();
		releaseRenderer();
		releaseRendererForWarpField();
	}

	void GpuMesh::create(size_t n)
	{
		// here we only release if memory not enough
		if (m_current_buffer_size < n*3*sizeof(PointType))
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
			// allocate slightly larger to avoid too dynamic updating.
			// cudaGraphicsGLRegisterBuffer is slow for large buffers.
			m_current_buffer_size = 3 * n * sizeof(PointType) * 1.5;
			glBufferData(GL_ARRAY_BUFFER, m_current_buffer_size, 0, GL_DYNAMIC_DRAW);
			if (n)// when n==0, it may crash.
			{
				cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_cuda_res, m_vbo_id,
					cudaGraphicsMapFlagsNone), "GpuMesh::create, cudaGraphicsGLRegisterBuffer");
			}
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
		m_num = n;
	}

	void GpuMesh::release()
	{
		wglMakeCurrent(g_hdc, g_glrc);
		if (m_vbo_id != 0)
			glDeleteBuffers(1, &m_vbo_id);
		if (m_cuda_res)
			cudaSafeCall(cudaGraphicsUnregisterResource(m_cuda_res), "GpuMesh::release(), unregister resouce");
		m_verts_d = nullptr;
		m_normals_d = nullptr;
		m_colors_d = nullptr;
		m_cuda_res = nullptr;
		m_vbo_id = 0;
		m_num = 0;
		m_current_buffer_size = 0;
	}

	void GpuMesh::copyFrom(GpuMesh& rhs)
	{
		create(rhs.num());

		rhs.lockVertsNormals();
		lockVertsNormals();

		cudaMemcpy(verts(), rhs.verts(), num() * 3 * sizeof(PointType), cudaMemcpyDeviceToDevice);

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
		cudaSafeCall(cudaGraphicsMapResources(1, &m_cuda_res, 0), "GpuMesh::lockVertsNormals(), 1");
		cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&m_verts_d, &num_bytes, m_cuda_res),
			"GpuMesh::lockVertsNormals() 2");
		m_normals_d = m_verts_d + m_num;
		m_colors_d = m_normals_d + m_num;
	}
	void GpuMesh::unlockVertsNormals()
	{
		if (m_verts_d == nullptr)
			return;
		cudaSafeCall(cudaGraphicsUnmapResources(1, &m_cuda_res, 0), "GpuMesh::unlockVertsNormals()");
		m_verts_d = nullptr;
		m_normals_d = nullptr;
		m_colors_d = nullptr;
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
		// to prevent dynamic allocation, we maintain a larger and larger render size.
		if (w > m_width || h > m_height)
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
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h,
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
				cudaGraphicsRegisterFlagsReadOnly), "GpuMesh::createRenderer, register gl buffer");
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

#ifdef ENABLE_SHOW_DEBUG
			g_testmesh = this;
			ShowWindow(g_hwnd, SHOW_OPENWINDOW);
#endif
		}
	}

	void GpuMesh::createRendererForWarpField(const WarpField* warpField)
	{
		if (warpField == nullptr)
			return;
		if (m_vbo_id_warpnodes == 0)
		{
			if (g_hdc == nullptr)
			{
				create_context_gpumesh();
				if (!wglMakeCurrent(g_hdc, g_glrc))
					throw std::exception("wglMakeCurrent error");
			}

			// create buffer object
			do{
				glGenBuffers(1, &m_vbo_id_warpnodes);
				CHECK_NOT_EQUAL(m_vbo_id_warpnodes, 0);
			} while (is_cuda_pbo_vbo_id_used_push_new(m_vbo_id_warpnodes));
			glBindBuffer(GL_ARRAY_BUFFER, m_vbo_id_warpnodes);

			int bytes_warp = WarpField::MaxNodeNum*WarpField::GraphLevelNum*(sizeof(float4)+
								+KnnK*sizeof(int)* 2);
			int bytes_corr = KINECT_WIDTH*KINECT_HEIGHT*(4*sizeof(float4)+2*sizeof(int));
			int bytes = max(bytes_warp, bytes_corr);
			glBufferData(GL_ARRAY_BUFFER, bytes, 0, GL_DYNAMIC_DRAW);
			cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_cuda_res_warp, m_vbo_id_warpnodes,
					cudaGraphicsMapFlagsNone), "GpuMesh::createRendererForWarpField, register gl buffer");
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
	}

	void GpuMesh::releaseRenderer()
	{
		if (m_render_fbo_id != 0)
		{
			wglMakeCurrent(g_hdc, g_glrc);
			glDeleteTextures(1, &m_render_texture_id);
			glDeleteRenderbuffers(1, &m_render_depth_id);
			glDeleteFramebuffers(1, &m_render_fbo_id);
			glDeleteBuffers(1, &m_render_fbo_pbo_id);
			cudaSafeCall(cudaGraphicsUnregisterResource(m_cuda_res_fbo), "GpuMesh::releaseRenderer");
		}
		m_width = 0;
		m_height = 0;
		m_render_fbo_id = 0;
		m_render_texture_id = 0;
		m_render_depth_id = 0;
		m_render_fbo_pbo_id = 0;
		m_cuda_res_fbo = nullptr;
	}

	void GpuMesh::releaseRendererForWarpField()
	{
		wglMakeCurrent(g_hdc, g_glrc);
		if (m_vbo_id_warpnodes != 0)
			glDeleteBuffers(1, &m_vbo_id_warpnodes);
		if (m_cuda_res)
			cudaSafeCall(cudaGraphicsUnregisterResource(m_cuda_res_warp), "GpuMesh::releaseRendererForWarpField");
		m_cuda_res_warp = nullptr;
		m_vbo_id_warpnodes = 0;
	}

	void GpuMesh::renderToImg(const Camera& camera, LightSource light, ColorMap& img, 
		const Param& param, const WarpField* warpField, 
		const MapArr* vmap_live, const MapArr* vmap_warp,
		const MapArr* nmap_live, const MapArr* nmap_warp, 
		GpuMesh* canoMesh, 
		const float3* canoPosActive,
		const KnnIdx* knnIdxActiveView,
		const Intr* intr, 
		bool warp_nodes,
		bool norigid)
	{
		if (!wglMakeCurrent(g_hdc, g_glrc))
			throw std::exception("wglMakeCurrent error");
		if (vmap_live || vmap_warp)
		{
			if (vmap_warp == nullptr || vmap_live == nullptr)
				throw std::exception("A pair of vmap must be provided in GpuMesh::renderToImg()");
			if (vmap_live->cols() != KINECT_WIDTH || vmap_live->rows() != KINECT_HEIGHT
				|| vmap_warp->cols() != KINECT_WIDTH || vmap_warp->rows() != KINECT_HEIGHT)
				throw std::exception("not supported vmap size in GpuMesh::renderToImg()");
		}
		const int width = std::lroundf(abs(camera.getViewPortRight() - camera.getViewPortLeft()));
		const int height = std::lroundf(abs(camera.getViewPortBottom() - camera.getViewPortTop()));

		if (param.view_show_mesh)
			createRenderer(width, height);
		if (param.view_show_nodes || param.view_show_graph || param.view_show_corr)
			createRendererForWarpField(warpField);

		img.create(height, width);

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_render_fbo_id);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		if (m_show_color)
		{
			glDisable(GL_LIGHTING);
			glDisable(GL_COLOR_MATERIAL);
		}
		else
		{
			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT0);
			glEnable(GL_COLOR_MATERIAL);
		}
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		glEnable(GL_POINT_SPRITE_ARB);
		glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1., 1.);

		ldp::Float3 sv = ldp::Float3(light.diffuse.x,
			light.diffuse.y, light.diffuse.z)*camera.getScalar();
		glLightfv(GL_LIGHT0, GL_DIFFUSE, sv.ptr());
		ldp::Float3 sa = 0.f;
		glLightfv(GL_LIGHT0, GL_AMBIENT, &light.amb.x);
		glLightfv(GL_LIGHT0, GL_SPECULAR, &light.spec.x);

		camera.apply();

		bool showColorVert = warpField && param.view_show_nodes && canoMesh;
		if (showColorVert)
			showColorVert = (warpField->getActiveVisualizeNodeId() >= 0);

		if (norigid && warpField)
		{
			Tbx::Transfo T = warpField->get_rigidTransform().fast_invert().transpose();
			glMultMatrixf(&T[0]);
		}

		// draw mesh vertices
		if (param.view_show_mesh)
		{
			if (showColorVert)
				update_color_buffer_by_warpField(warpField, canoMesh);

			unlockVertsNormals();
			glBindBuffer(GL_ARRAY_BUFFER, m_vbo_id);
			glVertexPointer(3, GL_FLOAT, sizeof(PointType), 0);
			glEnableClientState(GL_VERTEX_ARRAY);
			if (showColorVert || m_show_color)
			{
				size_t shift1 = m_num*sizeof(PointType)*2;
				glColorPointer(3, GL_FLOAT, sizeof(PointType), (GLvoid*)shift1);
				glEnableClientState(GL_COLOR_ARRAY);
			}

			if (!m_show_color)
			{
				size_t shift = m_num*sizeof(PointType);
				glNormalPointer(GL_FLOAT, sizeof(PointType), (GLvoid*)shift);
				glEnableClientState(GL_NORMAL_ARRAY);
			}

			glColor3f(0.7, 0.7, 0.7);
			glDrawArrays(GL_TRIANGLES, 0, m_num);
			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_NORMAL_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		// draw wrap field nodes
		if (warpField && (param.view_show_nodes || param.view_show_graph))
		{
			float4* gldata = nullptr;
			size_t num_bytes = 0;
			cudaSafeCall(cudaGraphicsMapResources(1, &m_cuda_res_warp, 0),
				"GpuMesh::renderToImg::mapWarpFieldRes");
			cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gldata, &num_bytes, m_cuda_res_warp),
				"GpuMesh::renderToImg::cudaGraphicsResourceGetMappedPointer");
			copy_warp_node_to_gl_buffer(gldata, warpField, warp_nodes, param.graph_single_level);
			cudaSafeCall(cudaGraphicsUnmapResources(1, &m_cuda_res_warp, 0), 
				"GpuMesh::renderToImg::unMapWarpFieldRes");

			glDisable(GL_LIGHTING);
			glEnableClientState(GL_VERTEX_ARRAY);

			// draw level nodes
			ldp::Float3 colors[4] = {
				ldp::Float3(0, 1, 0),
				ldp::Float3(0, 1, 1),
				ldp::Float3(1, 1, 0),
				ldp::Float3(1, 0, 0)
			};

			if (param.view_show_nodes)
			{
				glBindBuffer(GL_ARRAY_BUFFER, m_vbo_id_warpnodes);
				g_shader_node->begin();
				g_shader_node->setUniform1f("pointScale", width / tanf(camera.getFov()
					*0.5f*(float)M_PI / 180.0f));

				for (int level = 0; level < WarpField::GraphLevelNum; level++)
				{
					g_shader_node->setUniform1f("pointRadius", 0.002*(level+1));
					glVertexPointer(3, GL_FLOAT, sizeof(float4), 
						(void*)(WarpField::MaxNodeNum*sizeof(float4)*level));
					glColor3fv(colors[level].ptr());
					glDrawArrays(GL_POINTS, 0, warpField->getNumNodesInLevel(level));
				}

				if (showColorVert)
				{
					int id = warpField->getActiveVisualizeNodeId();
					g_shader_node->setUniform1f("pointRadius", 0.0021);
					glColor3f(1, 0, 1);
					glVertexPointer(3, GL_FLOAT, sizeof(float4),
						(void*)(id*sizeof(float4)));
					glDrawArrays(GL_POINTS, 0, 1);
				}

				if (knnIdxActiveView && canoPosActive)
				{
					float3 canoPos = *canoPosActive;
					const KnnIdxType *knnPtr = (const KnnIdxType*)knnIdxActiveView;

					glDisable(GL_DEPTH_TEST);
					// render knn
					g_shader_node->setUniform1f("pointRadius", 0.0021);
					glColor3f(1, 0, 1);
					for (int k = 0; k < KnnK; k++)
					{
						if (knnPtr[k] < warpField->getNumNodesInLevel(0))
						{
							glVertexPointer(3, GL_FLOAT, sizeof(float4),
								(void*)(knnPtr[k] * sizeof(float4)));
							glDrawArrays(GL_POINTS, 0, 1);
						}
					}

					// render vert
					g_shader_node->setUniform1f("pointRadius", 0.002);
					glColor3f(1, 0, 0);
					glBegin(GL_POINTS);
					glVertex3f(canoPos.x, canoPos.y, canoPos.z);
					glEnd();
					glEnable(GL_DEPTH_TEST);
				}

				g_shader_node->end();
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}

			// draw edges
			if (param.view_show_graph)
			{
				glBindBuffer(GL_ARRAY_BUFFER, m_vbo_id_warpnodes);
				glVertexPointer(3, GL_FLOAT, sizeof(float4), (void*)0);	
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vbo_id_warpnodes);
				glColor3fv(colors[param.view_show_graph_level].ptr());
				glDrawElements(GL_LINES, warpField->getNumNodesInLevel(param.view_show_graph_level)
					*KnnK * 2, GL_UNSIGNED_INT,
					(void*)(WarpField::MaxNodeNum*WarpField::GraphLevelNum*sizeof(float4) +
					WarpField::MaxNodeNum * 2 * KnnK*sizeof(int)* param.view_show_graph_level)
					);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}
			glDisableClientState(GL_VERTEX_ARRAY);
		}

		// show correspondence
		if (vmap_live && param.view_show_corr && intr)
		{
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_NORMAL_ARRAY);
			float4* gldata = nullptr;
			size_t num_bytes = 0;
			cudaSafeCall(cudaGraphicsMapResources(1, &m_cuda_res_warp, 0), "GpuMesh::renderToImg::cudaGraphicsMapResources1");
			cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gldata, &num_bytes, m_cuda_res_warp),
				"GpuMesh::renderToImg::cudaGraphicsMapResources2");
			copy_maps_to_gl_buffer(*vmap_live, *vmap_warp, *nmap_live, *nmap_warp, gldata, param, *intr);
			cudaSafeCall(cudaGraphicsUnmapResources(1, &m_cuda_res_warp, 0),
				 "GpuMesh::renderToImg::cudaGraphicsMapResources3");

			const int n = vmap_live->rows() * vmap_live->cols();
			glBindBuffer(GL_ARRAY_BUFFER, m_vbo_id_warpnodes);
			glVertexPointer(3, GL_FLOAT, sizeof(float4), (void*)0);
			glNormalPointer(GL_FLOAT, sizeof(float4), (void*)(n * 2 * sizeof(float4)));
			glEnable(GL_COLOR_MATERIAL);
			glEnable(GL_LIGHTING);
			glColor3f(0.4, 0, 0);
			glDrawArrays(GL_POINTS, 0, n);
			glColor3f(0.4, 0.4, 0);
			glDrawArrays(GL_POINTS, n, n);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vbo_id_warpnodes);
			glColor3f(0.0, 0.6, 0.0);
			glDrawElements(GL_LINES, n*2, GL_UNSIGNED_INT, (void*)(n * 4 * sizeof(float4)));
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glDisableClientState(GL_NORMAL_ARRAY);
			glDisableClientState(GL_VERTEX_ARRAY);
		}// end if vmap_live

		glPopAttrib();
		// do not use it:
		// it is only useful when draw to screen
		// here we use FBO and CUDA
		// swap buffers seems quite slow.
		//SwapBuffers(g_hdc);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, m_render_fbo_pbo_id);
		glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, 0);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

		float4* gldata = nullptr;
		size_t num_bytes = 0;
		cudaSafeCall(cudaGraphicsMapResources(1, &m_cuda_res_fbo, 0), 
			"GpuMesh::renderToImg::cudaGraphicsMapResources4");
		cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gldata, &num_bytes, m_cuda_res_fbo)
			, "GpuMesh::renderToImg::cudaGraphicsMapResources5");
		copy_invert_y(gldata, img);
		cudaSafeCall(cudaGraphicsUnmapResources(1, &m_cuda_res_fbo, 0)
			, "GpuMesh::renderToImg::cudaGraphicsMapResources6");

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
		CHECK_GL_ERROR("renderToImg");
	}

	void GpuMesh::renderToDepth(const Camera& camera, DepthMap& img)
	{
		if (!wglMakeCurrent(g_hdc, g_glrc))
			throw std::exception("wglMakeCurrent error");
		const int width = std::lroundf(abs(camera.getViewPortRight() - camera.getViewPortLeft()));
		const int height = std::lroundf(abs(camera.getViewPortBottom() - camera.getViewPortTop()));

		createRenderer(width, height);
		img.create(height, width);

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_render_fbo_id);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		camera.apply();
		unlockVertsNormals();
		g_shader_depth->begin();
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo_id);

		glVertexPointer(3, GL_FLOAT, sizeof(PointType), 0);
		glDrawArrays(GL_TRIANGLES, 0, m_num);

		glBindBuffer(GL_PIXEL_PACK_BUFFER, m_render_fbo_pbo_id);
		glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, 0);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
		float4* gldata = nullptr;
		size_t num_bytes = 0;
		cudaSafeCall(cudaGraphicsMapResources(1, &m_cuda_res_fbo, 0)
			, "GpuMesh::renderToDepth 1");
		cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gldata, &num_bytes, m_cuda_res_fbo)
			, "GpuMesh::renderToDepth 2");
		const float s1 = 2.f*camera.getFrustumNear()*camera.getFrustumFar() /
			(camera.getFrustumNear() - camera.getFrustumFar());
		const float s2 = (camera.getFrustumNear() + camera.getFrustumFar()) /
			(camera.getFrustumNear() - camera.getFrustumFar());
		copy_gldepth_to_depthmap(gldata, img, s1, s2, camera.getFrustumNear());
		cudaSafeCall(cudaGraphicsUnmapResources(1, &m_cuda_res_fbo, 0)
			, "GpuMesh::renderToDepth 3");

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDisableClientState(GL_VERTEX_ARRAY);
		glPopAttrib();
		g_shader_depth->end();

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
		CHECK_GL_ERROR("renderToDepth");
	}

	void GpuMesh::renderToCanonicalMaps(const Camera& camera,
		GpuMesh* canoMesh, DeviceArray2D<float4>& vmap, 
		DeviceArray2D<float4>& nmap)
	{
		if (canoMesh->num() != m_num)
			throw std::exception("GpuMesh::renderToCanonicalMaps(): mesh size not matched!");
		if (!wglMakeCurrent(g_hdc, g_glrc))
			throw std::exception("wglMakeCurrent error");
		const int width = std::lroundf(abs(camera.getViewPortRight() - camera.getViewPortLeft()));
		const int height = std::lroundf(abs(camera.getViewPortBottom() - camera.getViewPortTop()));

		createRenderer(width, height);
		vmap.create(height, width);
		nmap.create(height, width);

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_render_fbo_id);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		unlockVertsNormals();
		canoMesh->unlockVertsNormals();

		g_shader_cano->begin();
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo_id);
		glVertexPointer(3, GL_FLOAT, sizeof(PointType), 0);
		glBindBuffer(GL_ARRAY_BUFFER, canoMesh->m_vbo_id);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, m_render_fbo_pbo_id);
		camera.apply();

		// draw cano vertices as colors
		glColorPointer(3, GL_FLOAT, sizeof(PointType), 0);
		glDrawArrays(GL_TRIANGLES, 0, m_num);
		// read the buffer
		{
			glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, 0);
			float4* gldata = nullptr;
			size_t num_bytes = 0;
			cudaSafeCall(cudaGraphicsMapResources(1, &m_cuda_res_fbo, 0),
				"GpuMesh::renderToCanonicalMaps 1");
			cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gldata, &num_bytes, m_cuda_res_fbo),
				"GpuMesh::renderToCanonicalMaps 2");
			copy_canoview(gldata, vmap);
			cudaSafeCall(cudaGraphicsUnmapResources(1, &m_cuda_res_fbo, 0),
				"GpuMesh::renderToCanonicalMaps 3");
		}
		g_shader_cano->end();

		// draw cano normals as colors
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		g_shader_cano->begin();
		glColorPointer(3, GL_FLOAT, sizeof(PointType), (void*)(canoMesh->num()*sizeof(PointType)));
		glDrawArrays(GL_TRIANGLES, 0, m_num);
		// read the buffer
		{
			glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, 0);
			float4* gldata = nullptr;
			size_t num_bytes = 0;
			cudaSafeCall(cudaGraphicsMapResources(1, &m_cuda_res_fbo, 0),
				"GpuMesh::renderToCanonicalMaps 4");
			cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&gldata, &num_bytes, m_cuda_res_fbo),
				"GpuMesh::renderToCanonicalMaps 5");
			copy_canoview(gldata, nmap);
			cudaSafeCall(cudaGraphicsUnmapResources(1, &m_cuda_res_fbo, 0),
				"GpuMesh::renderToCanonicalMaps 6");
		}

		g_shader_cano->end();
		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
		glPopAttrib();
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
		CHECK_GL_ERROR("renderToCanonicalMaps");
	}
}