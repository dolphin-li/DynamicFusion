#include "kinect_util.h"
#include "ObjMesh.h"
namespace dfusion
{
	void mapsToObj(ObjMesh& mesh, const dfusion::MapArr& vmap, const dfusion::MapArr& nmap)
	{
		int w = vmap.cols();
		int h = vmap.rows() / 3;
		int n = w*h;
		std::vector<float> vhost(n * 3), nhost(n * 3);
		vmap.download(vhost.data(), w*sizeof(float));
		nmap.download(nhost.data(), w*sizeof(float));

		mesh.clear();
		mesh.vertex_list.resize(n);
		mesh.vertex_normal_list.resize(n);

		for (int i = 0; i < n; i++)
		{
			mesh.vertex_list[i][0] = vhost[i];
			mesh.vertex_list[i][1] = vhost[i + n];
			mesh.vertex_list[i][2] = vhost[i + n * 2];
			mesh.vertex_normal_list[i][0] = nhost[i];
			mesh.vertex_normal_list[i][1] = nhost[i + n];
			mesh.vertex_normal_list[i][2] = nhost[i + n * 2];
		}
	}
}