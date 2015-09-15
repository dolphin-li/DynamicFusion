#pragma once
#include "VolumeData.h"
#include "ObjMesh.h"
namespace mpu
{
	class MarchingCube
	{
	public:
		MarchingCube();
		~MarchingCube();

		void run(const VolumeData& data, ObjMesh& mesh);
		void run(const VolumeData& data, ObjMesh& mesh, ldp::Int3 begin, ldp::Int3 end);
		void run(const VolumeData& data, ObjMesh& mesh, kdtree::AABB box);
	protected:
		struct GridCell 
		{
			// input field for marching cube
			ldp::Float3 p[8];//position of each corner of the grid in world space
			float val[8];	//value of the function at this grid corner

			// output field for marching cube
			ldp::Int3 out_tri[10];
			ldp::Float3 out_vert[15];
			int n_out_tri;
			int n_out_vert;

			// tmp field
			ldp::Float3 tmp_vert[12];
			char tmp_map[12];
		};

		// given a grid cell, returns the set of triangles that approximates the region where val == 0.
		// NOTE:
		//		preallocate Triangles[10] and Vertices [15]
		void polygonize(GridCell &Grid);
	private:
	};
}