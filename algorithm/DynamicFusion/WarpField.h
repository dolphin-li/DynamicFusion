#pragma once

#include "definations.h"
namespace dfusion
{
	class WarpField
	{
	public:
		WarpField();
		~WarpField();

		// estimate from 2 maps
		void estimateRigid(const MapArr& v0, const MapArr& n0, const MapArr& v1, const MapArr& n1);


		Tbx::Transfo get_rigidTransform()const{ return m_rigidTransform; }
		void set_rigidTransform(Tbx::Transfo T){ m_rigidTransform = T; }
	private:
		Tbx::Transfo m_rigidTransform;
	};

}