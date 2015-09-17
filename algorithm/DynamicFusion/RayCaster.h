#pragma once

#include "definations.h"

class Camera;

namespace dfusion
{
	class TsdfVolume;
	class RayCaster
	{
	public:
		RayCaster();
		~RayCaster();

		void init(const TsdfVolume& v);

		void setCamera(const Camera& cam);

		// host_gray_buffer: pre-allocated, size = viewport width*height defined by the camera
		// host_step: strides of each row in bytes
		void shading(LightSource light, ColorMap& colorMap, bool show_normal_map=false);

		void clear();
	protected:
		void raycast();
	private:
		const TsdfVolume* m_volume;
		Intr m_intr;
		Mat33 m_Rc2v;
		float3 m_tc2v;
		Mat33 m_Rv2w; 
		float3 m_tv2w;
		
		MapArr m_vmap;
		MapArr m_nmap;
	};

}