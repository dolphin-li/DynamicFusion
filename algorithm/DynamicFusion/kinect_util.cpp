#include "kinect_util.h"
#include "ObjMesh.h"
namespace dfusion
{
	void mapsToObj(ObjMesh& mesh, const dfusion::MapArr& vmap, const dfusion::MapArr& nmap)
	{
		int w = vmap.cols();
		int h = vmap.rows();
		int n = w*h;
		std::vector<float4> vhost(n), nhost(n);
		vmap.download(vhost.data(), w*sizeof(float4));
		nmap.download(nhost.data(), w*sizeof(float4));

		mesh.clear();
		mesh.vertex_list.resize(n);
		mesh.vertex_normal_list.resize(n);

		for (int i = 0; i < n; i++)
		{
			mesh.vertex_list[i][0] = vhost[i].x;
			mesh.vertex_list[i][1] = vhost[i].y;
			mesh.vertex_list[i][2] = vhost[i].z;
			mesh.vertex_normal_list[i][0] = nhost[i].x;
			mesh.vertex_normal_list[i][1] = vhost[i].y;
			mesh.vertex_normal_list[i][2] = vhost[i].z;
		}
	}
}