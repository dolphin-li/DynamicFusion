#include "RayCaster.h"
#include "TsdfVolume.h"
#include "fmath.h"
#include "Camera.h"
namespace dfusion
{
	RayCaster::RayCaster()
	{
		m_volume = nullptr;
	}

	RayCaster::~RayCaster()
	{
	}

	void RayCaster::init(const TsdfVolume& V)
	{
		m_volume = &V;
	}

	void RayCaster::setCamera(const Camera& cam)
	{
		release_assert(m_volume);
		const float l = cam.getViewPortLeft();
		const float r = cam.getViewPortRight();
		const float t = cam.getViewPortTop();
		const float b = cam.getViewPortBottom();
		const float	f = (b-t) * 0.5f / tanf(cam.getFov() * fmath::DEG_TO_RAD * 0.5f);

		// origion
		ldp::Float3 ori(m_volume->getOrigion().x, m_volume->getOrigion().y, m_volume->getOrigion().z);
		ldp::Mat4f Omat = ldp::Mat4f().eye();
		Omat.setTranslationPart(ori);

		// -y, -z
		ldp::Mat4f nynz = ldp::Mat4f().eye();
		nynz(1, 1) = nynz(2, 2) = -1;

		ldp::Mat4f volume2world = nynz * cam.getModelViewMatrix() * Omat;
		ldp::Mat4f camera2volume = volume2world.inv();

		// camera intrinsic
		m_intr.cx = (r + l)*0.5f - 0.5f;
		m_intr.cy = (b + t)*0.5f - 0.5f;
		m_intr.fx = f;
		m_intr.fy = f;

		// volume to world
		// NOTE: in mat33, each vec is a row-vec
		for (int x = 0; x < 3; x++)
		{
			m_Rv2w.data[x].x = volume2world(x, 0);
			m_Rv2w.data[x].y = volume2world(x, 1);
			m_Rv2w.data[x].z = volume2world(x, 2);
		}
		m_tv2w.x = volume2world(0, 3);
		m_tv2w.y = volume2world(1, 3);
		m_tv2w.z = volume2world(2, 3);

		// camera to volume
		for (int x = 0; x < 3; x++)
		{
			m_Rc2v.data[x].x = camera2volume(x, 0);
			m_Rc2v.data[x].y = camera2volume(x, 1);
			m_Rc2v.data[x].z = camera2volume(x, 2);
		}
		m_tc2v.x = camera2volume(0, 3);
		m_tc2v.y = camera2volume(1, 3);
		m_tc2v.z = camera2volume(2, 3);

		// allocate 2d buffers
		m_vmap.create(std::lroundf(b - t) * 3, std::lroundf(r - l));
		m_nmap.create(m_vmap.rows(), m_vmap.cols());	
	}

	// host_gray_buffer: pre-allocated, size = viewport width*height defined by the camera
	void RayCaster::shading(LightSource light, ColorMap& colorMap, bool show_normal_map)
	{
		release_assert(m_volume);

		colorMap.create(m_vmap.rows() / 3, m_vmap.cols());

		// 1. ray casting
		raycast();

		// 2. shading
		if (show_normal_map)
		{
			Mat33 R = m_Rc2v;
			generateNormalMap(m_nmap, colorMap, R);
		}
		else
		{
			generateImage(m_vmap, m_nmap, colorMap, light);
		}
	}

	void RayCaster::clear()
	{
		m_volume = nullptr;
		m_vmap.release();
		m_nmap.release();
	}

}