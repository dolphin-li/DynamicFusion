#pragma once
#include "nanoflann.hpp"
#include "ldp_basic_vec.h"
namespace ldp
{

	namespace kdtree
	{
		template <typename T>
		class PointTree
		{
		public:
			class Point
			{
			public:
				int idx;
				ldp::ldp_basic_vec3<T> p;

				Point() :idx(-1), p(0){}
				Point(const Point& r) :idx(r.idx), p(r.p){}
				Point(ldp::ldp_basic_vec3<T>& v, int i = -1) : p(v), idx(i){}
			};
		private:
			template <typename T>
			struct PointCloud
			{
				std::vector<Point>  pts;

				// Must return the number of data points
				inline size_t kdtree_get_point_count() const { return pts.size(); }

				// Returns the distance between the vector "p1[0:size-1]" 
				// and the data point with index "idx_p2" stored in the class:
				inline T kdtree_distance(const T *p1, const size_t idx_p2, size_t size) const
				{
					const T d0 = p1[0] - pts[idx_p2].p[0];
					const T d1 = p1[1] - pts[idx_p2].p[1];
					const T d2 = p1[2] - pts[idx_p2].p[2];
					return d0*d0 + d1*d1 + d2*d2;
				}

				inline T kdtree_get_pt(const size_t idx, int dim) const
				{
					return pts[idx].p[dim];
				}

				// Optional bounding-box computation: return false to default to a standard bbox computation loop.
				//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
				//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
				template <class BBOX>
				bool kdtree_get_bbox(BBOX &bb) const { return false; }

			};
			typedef nanoflann::KDTreeSingleIndexAdaptor<
				nanoflann::L2_Simple_Adaptor<T, PointCloud<T> >,
				PointCloud<T>,
				3 /* dim */
			> my_kd_tree_simple_t;

			PointCloud<T> m_points;
			my_kd_tree_simple_t* m_tree;
		public:
			PointTree()
			{
				m_tree = 0;
			}
			~PointTree()
			{
				clear();
			}
			void clear()
			{
				if (m_tree)
					delete m_tree;
				m_tree = 0;
				m_points.pts.clear();
			}
			PointTree& cloneFrom(const PointTree& rhs)
			{
				clear();
				m_points = rhs.m_points;
				m_tree = new my_kd_tree_simple_t(3 /*dim*/, m_points,
					nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
				m_tree->buildIndex();
				return *this;
			}

			void build(const std::vector<Point>& points)
			{
				clear();
				m_points.pts.assign(points.begin(), points.end());
				m_tree = new my_kd_tree_simple_t(3 /*dim*/, m_points,
					nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
				m_tree->buildIndex();
			}

			bool isValid()
			{
				return m_points.pts.size() != 0;
			}

			Point nearestPoint(Point p, T& dist)const
			{
				size_t index;
				T out_dist_sqr = T(0);
				nanoflann::KNNResultSet<T> resultSet(1);
				resultSet.init(&index, &out_dist_sqr);
				m_tree->findNeighbors(resultSet, p.p.ptr(), nanoflann::SearchParams(10));

				dist = sqrt(out_dist_sqr);
				return m_points.pts[index];
			}

			void kNearestPoints(Point p, int* idx, T* dist, int k)
			{
				nanoflann::KNNResultSet<T> resultSet(k);
				std::vector<size_t> tmpIdx(k);
				resultSet.init(tmpIdx.data(), dist);
				m_tree->findNeighbors(resultSet, p.p.ptr(), nanoflann::SearchParams(10));

				for (int i = 0; i < k; i++)
				{
					dist[i] = sqrt(dist[i]);
					idx[i] = tmpIdx[i];
				}
			}

			void pointInSphere(Point c, T radius, std::vector<std::pair<size_t, T>>& indices_dists)
			{
				nanoflann::RadiusResultSet<T> resultSet(radius, indices_dists);
				m_tree->findNeighbors(resultSet, c.p.ptr(), nanoflann::SearchParams(10));
			}
		};
	}
}