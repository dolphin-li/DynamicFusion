#pragma once

#include "definations.h"
namespace dfusion
{
	class GpuMesh;
	class WarpField
	{
	public:
		WarpField();
		~WarpField();

		void warp(GpuMesh& src, GpuMesh& dst);

		Tbx::Transfo get_rigidTransform()const{ return m_rigidTransform; }
		void set_rigidTransform(Tbx::Transfo T){ m_rigidTransform = T; }
	private:
		Tbx::Transfo m_rigidTransform;
	};

}