#include "CpuGaussNewton.h"
#include "WarpField.h"
#include "LMSolver.h"
#include <helper_math.h>
#include <set>
#include "ldpdef.h"
namespace dfusion
{
	inline float3 read_float3_from_4(float4 p)
	{
		return make_float3(p.x, p.y, p.z);
	}
	inline float norm(float3 a)
	{
		return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
	}


	typedef WarpField::KnnIdx KnnIdx;
	struct EigenContainter
	{
		typedef float real;
		typedef Eigen::Matrix<real, -1, 1> Vec;
		typedef Eigen::SparseMatrix<real, Eigen::ColMajor> SpMat;

		// 6-twist of exponential map of dual-quaternion
		Vec x_;
		SpMat jac_, jact_;

		std::vector<KnnIdx> vmapKnn_;
		std::vector<KnnIdx> nodesKnn_;
		std::vector<float4> nodesVw_;
		std::vector<float4> vmap_cano_;
		std::vector<float4> nmap_cano_;
		std::vector<float4> vmap_warp_;
		std::vector<float4> nmap_warp_;
		std::vector<float4> vmap_live_;
		std::vector<float4> nmap_live_;
		std::vector<int> map_c2l_corr_;
		std::vector<int> row_index_of_pixel_;
		std::vector<int> coo_pos_of_pixel_;
		int nPixel_rows_;
		int nPixel_cooPos_;
		int imgWidth_;
		int imgHeight_;
		Param param_;
		Intr intr_;
		Tbx::Transfo Tlw_;
		std::vector<Eigen::Triplet<real>> m_cooSys;

		void CalcCorr()
		{
			map_c2l_corr_.resize(imgWidth_*imgHeight_);
			std::fill(map_c2l_corr_.begin(), map_c2l_corr_.end(), -1);

			for (int y = 0; y < imgHeight_; y++)
			{
				for (int x = 0; x < imgWidth_; x++)
				{
					float3 pwarp = read_float3_from_4(vmap_warp_[y*imgWidth_ + x]);
					float3 nwarp = read_float3_from_4(nmap_warp_[y*imgWidth_ + x]);		
					
					if (isnan(nwarp.x))
						continue;

					float3 uvd = intr_.xyz2uvd(pwarp);
					int2 ukr = make_int2(uvd.x+0.5, uvd.y+0.5);

					// we use opengl coordinate, thus world.z should < 0
					if (ukr.x < 0 || ukr.y < 0 || ukr.x >= imgWidth_ || ukr.y >= imgHeight_ || pwarp.z >= 0)
						continue;

					float3 plive = read_float3_from_4(vmap_live_[ukr.y*imgWidth_ + ukr.x]);
					float3 nlive = read_float3_from_4(nmap_live_[ukr.y*imgWidth_ + ukr.x]);
					if (isnan(nlive.x))
						continue;

					float dist = norm(pwarp - plive);
					if (dist > param_.fusion_nonRigid_distThre)
						continue;

					float sine = norm(cross(nwarp, nlive));
					if (sine >= param_.fusion_nonRigid_angleThreSin)
						continue;

					map_c2l_corr_[y*imgWidth_ + x] = ukr.y*imgWidth_ + ukr.x;
				}// x
			}// y
		}

		real Optimize(Vec& xStart, int nMaxIter, bool showInfo = true)
		{
			//define jacobi structure
			DefineJacobiStructure(jac_, jact_);

			SpMat JacTJac;
			Vec fx(jac_.rows()), h(jac_.cols()), g(jac_.cols()), fx1(jac_.rows());

			//define structure of J'J
			JacTJac = jact_ * jac_;
			Eigen::SimplicialCholesky<SpMat> solver;
			solver.analyzePattern(JacTJac.triangularView<Eigen::Lower>());

			//Gauss-Newton Optimization
			for (int iter = 0; iter<nMaxIter; iter++)
			{
				CalcJacobiFunc(xStart, jac_, jact_);	//J
				JacTJac = jact_ * jac_;//J'J
				CalcEnergyFunc(xStart, fx);	//f

				//solve: J'J h =  - J' f(x)
				g = jact_ * (-fx);
				solver.factorize(JacTJac.triangularView<Eigen::Lower>());
				h = solver.solve(g);

				real normv = xStart.norm();
				double old_energy = fx.dot(fx);
				for (real alpha = 1; alpha > 1e-15; alpha *= 0.5)
				{
					Vec x = xStart + h;
					CalcEnergyFunc(x, fx1);	//f
					double new_energy = fx1.dot(fx1);
					if (new_energy > old_energy)
						h = h * 0.5;
					if (new_energy < old_energy)
					{
						xStart = x;
						break;
					}
				}
				real normh = h.norm();

				if (showInfo)
					printf("Gauss-Newton: %d -- %f = %f/%f: %f\n", iter, normh / (1e-6+normv), 
					normh, normv, old_energy);

				if (normh < (normv + real(1e-6)) * real(1e-6))
					break;
			}

			return fx.dot(fx);
		}

		inline real data_term_penalty(real v)const
		{
			// the penalty function is ||v||^2, thus a single term is v.
			return v;
		}
		
		inline real data_term_grad(real f)const
		{
			// the penalty function single term is f, thus the gradient is 2f
			return 1;
		}
		
		inline Tbx::Vec3 reg_term_penalty(Tbx::Vec3 v)const
		{
			return v;
		}
		
		inline Tbx::Mat3 reg_term_grad(Tbx::Vec3 f)const
		{
			return Tbx::Mat3::identity();
		}
		
		inline Tbx::Transfo outer_product(Tbx::Vec3 n, Tbx::Point3 v)const
		{
			return Tbx::Transfo(
				n.x*v.x, n.x*v.y, n.x*v.z, n.x,
				n.y*v.x, n.y*v.y, n.y*v.z, n.y,
				n.z*v.x, n.z*v.y, n.z*v.z, n.z,
				0, 0, 0, 0
				);
		}

		inline Tbx::Dual_quat_cu p_qk_p_alpha_func(Tbx::Dual_quat_cu dq, int i)
		{
			Tbx::Vec3 t, r;
			float b, c, n;
			Tbx::Quat_cu q0(0,0,0,0), q1=dq.get_non_dual_part();
			switch (i)
			{
			case 0:
				dq.to_twist(r, t);
				n = r.norm();
				if (n > std::numeric_limits<real>::epsilon())
				{
					b = sin(n) / n;
					c = (cos(n) - b) / (n*n);
					q0.coeff0 = -r.x * b;
					q0.coeff1 = b + r.x*r.x*c;
					q0.coeff2 = r.x*r.y*c;
					q0.coeff3 = r.x*r.z*c;
				}
				else
				{
					q0.coeff0 = 0;
					q0.coeff1 = 1;
					q0.coeff2 = 0;
					q0.coeff3 = 0;
				}

				q1.coeff0 = (t.x * q0.coeff1 + t.y * q0.coeff2 + t.z * q0.coeff3) * (-0.5);
				q1.coeff1 = (t.x * q0.coeff0 + t.y * q0.coeff3 - t.z * q0.coeff2) * 0.5;
				q1.coeff2 = (-t.x * q0.coeff3 + t.y * q0.coeff0 + t.z * q0.coeff1) * 0.5;
				q1.coeff3 = (t.x * q0.coeff2 - t.y * q0.coeff1 + t.z * q0.coeff0) * 0.5;
				return Tbx::Dual_quat_cu(q0, q1);
			case 1:
				dq.to_twist(r, t);
				n = r.norm(); 
				if (n > std::numeric_limits<real>::epsilon())
				{
					b = sin(n) / n;
					c = (cos(n) - b) / (n*n);
					q0.coeff0 = -r.y * b;
					q0.coeff1 = r.y*r.x*c;
					q0.coeff2 = b + r.y*r.y*c;
					q0.coeff3 = r.y*r.z*c;
				}
				else
				{
					q0.coeff0 = 0;
					q0.coeff1 = 0;
					q0.coeff2 = 1;
					q0.coeff3 = 0;
				}

				q1.coeff0 = (t.x * q0.coeff1 + t.y * q0.coeff2 + t.z * q0.coeff3) * (-0.5);
				q1.coeff1 = (t.x * q0.coeff0 + t.y * q0.coeff3 - t.z * q0.coeff2) * 0.5;
				q1.coeff2 = (-t.x * q0.coeff3 + t.y * q0.coeff0 + t.z * q0.coeff1) * 0.5;
				q1.coeff3 = (t.x * q0.coeff2 - t.y * q0.coeff1 + t.z * q0.coeff0) * 0.5;
				return Tbx::Dual_quat_cu(q0, q1);
			case 2:
				dq.to_twist(r, t);
				n = r.norm();
				if (n > std::numeric_limits<real>::epsilon())
				{
					b = sin(n) / n;
					c = (cos(n) - b) / (n*n);

					q0.coeff0 = -r.z * b;
					q0.coeff1 = r.z*r.x*c;
					q0.coeff2 = r.z*r.y*c;
					q0.coeff3 = b + r.z*r.z*c;
				}
				else
				{
					q0.coeff0 = 0;
					q0.coeff1 = 0;
					q0.coeff2 = 0;
					q0.coeff3 = 1;
				}

				q1.coeff0 = ( t.x * q0.coeff1 + t.y * q0.coeff2 + t.z * q0.coeff3) * (-0.5);
				q1.coeff1 = ( t.x * q0.coeff0 + t.y * q0.coeff3 - t.z * q0.coeff2) * 0.5;
				q1.coeff2 = (-t.x * q0.coeff3 + t.y * q0.coeff0 + t.z * q0.coeff1) * 0.5;
				q1.coeff3 = ( t.x * q0.coeff2 - t.y * q0.coeff1 + t.z * q0.coeff0) * 0.5;
				return Tbx::Dual_quat_cu(q0, q1);
			case 3:
				return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff1, q1.coeff0, -q1.coeff3, q1.coeff2))*0.5;
			case 4:
				return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff2, q1.coeff3, q1.coeff0, -q1.coeff1))*0.5;
			case 5:
				return Tbx::Dual_quat_cu(q0, Tbx::Quat_cu(-q1.coeff3, -q1.coeff2, q1.coeff1, q1.coeff0))*0.5;
			default:
				throw std::exception("p_qk_p_alpha_func: out of range");
				break;
			}
		}

		inline float trace_AtB(Tbx::Transfo A, Tbx::Transfo B)
		{
			float sum = 0;
			for (int i = 0; i < 16; i++)
				sum += A[i] * B[i];
			return sum;
		}

		inline Tbx::Transfo p_SE3_p_dq_func(Tbx::Dual_quat_cu dq, int i)
		{
			Tbx::Quat_cu q0 = dq.get_non_dual_part();
			Tbx::Quat_cu q1 = dq.get_dual_part();
			real x0 = q0.i(), y0 = q0.j(), z0 = q0.k(), w0 = q0.w();
			real x1 = q1.i(), y1 = q1.j(), z1 = q1.k(), w1 = q1.w();
			switch (i)
			{
			case 0:
				return Tbx::Transfo(
					0, -z0, y0, x1,
					z0, 0, -x0, y1,
					-y0, x0, 0, z1,
					0, 0, 0, 0) * 2;
			case 1:
				return Tbx::Transfo(
					0, y0, z0, -w1,
					y0, -2*x0, -w0, -z1,
					z0, w0, -2*x0, y1,
					0, 0, 0, 0) * 2;
			case 2:
				return Tbx::Transfo(
					-2*y0, x0, w0, z1,
					x0, 0, z0, -w1,
					-w0, z0, -2*y0, -x1,
					0, 0, 0, 0) * 2;
			case 3:
				return Tbx::Transfo(
					-2*z0, -w0, x0, -y1,
					w0, -2*z0, y0, x1,
					x0, y0, 0, -w1,
					0, 0, 0, 0) * 2;
			case 4:
				return Tbx::Transfo(
					0, 0, 0, -x0,
					0, 0, 0, -y0,
					0, 0, 0, -z0,
					0, 0, 0, 0) * 2;
			case 5:
				return Tbx::Transfo(
					0, 0, 0, w0,
					0, 0, 0, z0,
					0, 0, 0, -y0,
					0, 0, 0, 0) * 2;
			case 6:
				return Tbx::Transfo(
					0, 0, 0, -z0,
					0, 0, 0, w0,
					0, 0, 0, x0,
					0, 0, 0, 0) * 2;
			case 7:
				return Tbx::Transfo(
					0, 0, 0, y0,
					0, 0, 0, -x0,
					0, 0, 0, w0,
					0, 0, 0, 0) * 2;
			default:
				throw std::exception("index out of range");
				return Tbx::Transfo::identity();
			}
		}

		inline Tbx::Transfo p_SE3_p_alpha_func(Tbx::Dual_quat_cu dq, int i)
		{
			Tbx::Transfo T = Tbx::Transfo::empty();
			Tbx::Dual_quat_cu p_dq_p_alphai = p_qk_p_alpha_func(dq, i);
			for (int j = 0; j < 8; j++)
			{
				T = T + p_SE3_p_dq_func(dq, j)*p_dq_p_alphai[j];
			}
			return T;
		}

		inline WarpField::IdxType& knn_k(WarpField::KnnIdx& knn, int k)const
		{
			return ((WarpField::IdxType*)(&knn))[k];
		}

		void DefineJacobiStructure(SpMat& jac, SpMat& jact)
		{
			enum {VarPerNode = 6};
			const int nNodes = x_.size() / 6;
			const int nPixel = imgHeight_ * imgWidth_;

			m_cooSys.clear();
			coo_pos_of_pixel_.resize(nPixel);
			row_index_of_pixel_.resize(nPixel);
			std::fill(row_index_of_pixel_.begin(), row_index_of_pixel_.end(), -1);
			std::fill(coo_pos_of_pixel_.begin(), coo_pos_of_pixel_.end(), -1);

			// data term
			int nRow = 0;
			for (int iPixel = 0; iPixel < nPixel; iPixel++)
			{
				int corrPixel = map_c2l_corr_[iPixel];
				if (corrPixel < 0)
					continue;

				coo_pos_of_pixel_[iPixel] = m_cooSys.size();

				bool valid = false;
				KnnIdx knn = vmapKnn_[iPixel];
				for (int k = 0; k < WarpField::KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						valid = true;
						for (int t = 0; t < VarPerNode; t++)
						{
							m_cooSys.push_back(Eigen::Triplet<real>(nRow, 
							knnNodeId * VarPerNode + t, 0));
						}
					}
				}
				if (valid)
					row_index_of_pixel_[iPixel] = nRow++;
			}// end for iPixel
			nPixel_rows_ = nRow;
			nPixel_cooPos_ = m_cooSys.size();

			// reg term
			for (int iNode = 0; iNode < nNodes; iNode++)
			{
				KnnIdx knn = nodesKnn_[iNode];
				for (int k = 0; k < WarpField::KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						for (int ixyz = 0; ixyz < 3; ixyz++, nRow++)
						for (int t = 0; t < VarPerNode; t++)
						{
							m_cooSys.push_back(Eigen::Triplet<real>(nRow, iNode * VarPerNode + t, 0));
							m_cooSys.push_back(Eigen::Triplet<real>(nRow, knnNodeId * VarPerNode + t, 0));
						}
					}
				}
			}// end for iNode

			jac.resize(nRow, nNodes * VarPerNode);
			if (m_cooSys.size())
				jac.setFromTriplets(m_cooSys.begin(), m_cooSys.end());
			jact = jac.transpose();
		}

		void CalcEnergyFunc(const Eigen::VectorXf& x, Eigen::VectorXf& f)const
		{
			int nNodes = x.size() / 6;

			// data term
#pragma omp parallel for
			for (int iPixel = 0; iPixel < map_c2l_corr_.size(); iPixel++)
			{
				int nRow = row_index_of_pixel_[iPixel];
				if (nRow < 0)
					continue;

				int corrPixel = map_c2l_corr_[iPixel];

				Tbx::Vec3 v = convert(read_float3_from_4(vmap_cano_[iPixel]));
				Tbx::Vec3 n = convert(read_float3_from_4(nmap_cano_[iPixel]));
				Tbx::Vec3 vl = convert(read_float3_from_4(vmap_live_[corrPixel]));

				KnnIdx knn = vmapKnn_[iPixel];
				Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
				for (int k = 0; k < WarpField::KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						Tbx::Dual_quat_cu dq;
						Tbx::Vec3 r(x[knnNodeId * 6], x[knnNodeId * 6 + 1], x[knnNodeId * 6 + 2]);
						Tbx::Vec3 t(x[knnNodeId * 6 + 3], x[knnNodeId * 6 + 4], x[knnNodeId * 6 + 5]);
						Tbx::Vec3 nodesV = convert(read_float3_from_4(nodesVw_[knnNodeId]));
						float nodesW = nodesVw_[knnNodeId].w; 
						dq.from_twist(r, t);
						// note: we store inv radius as vw.w, thus using * instead of / here
						dq_blend = dq_blend + dq*exp(-(v - nodesV).dot(v - nodesV)*(2 * nodesW*nodesW));
					}
				}

				dq_blend.normalize();
				v = Tlw_*(dq_blend.transform(Tbx::Point3(v)));
				n = Tlw_*(dq_blend.rotate(n));
				f[nRow] = data_term_penalty(n.dot(v - vl));
			}// end for iPixel

			// reg term
			int nRow = nPixel_rows_;
			const float lambda = sqrt(param_.fusion_lambda);
			for (int iNode = 0; iNode < nNodes; iNode++)
			{
				KnnIdx knn = nodesKnn_[iNode];
				Tbx::Dual_quat_cu dqi;
				Tbx::Vec3 ri(x[iNode * 6], x[iNode * 6 + 1], x[iNode * 6 + 2]);
				Tbx::Vec3 ti(x[iNode * 6 + 3], x[iNode * 6 + 4], x[iNode * 6 + 5]);
				dqi.from_twist(ri, ti);

				for (int k = 0; k < WarpField::KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						Tbx::Vec3 rj(x[knnNodeId * 6], x[knnNodeId * 6 + 1], x[knnNodeId * 6 + 2]);
						Tbx::Vec3 tj(x[knnNodeId * 6 + 3], x[knnNodeId * 6 + 4], x[knnNodeId * 6 + 5]);
						real alpha_ij = sqrt(max(1 / nodesVw_[iNode].w, 1 / nodesVw_[knnNodeId].w));

						Tbx::Dual_quat_cu dqj;
						dqj.from_twist(rj, tj);
						Tbx::Vec3 vj = convert(read_float3_from_4(nodesVw_[knnNodeId]));
						Tbx::Vec3 val = dqi.transform(Tbx::Point3(vj)) - dqj.transform(Tbx::Point3(vj));
						val = reg_term_penalty(val);
						f[nRow++] = val.x * lambda * alpha_ij;
						f[nRow++] = val.y * lambda * alpha_ij;
						f[nRow++] = val.z * lambda * alpha_ij;
					}
				}
			}// end for iNode
		}

		int FindColIndices(const SpMat& A, int cid, int* vidx, int* ridx)
		{
			int ns = A.outerIndexPtr()[cid], ne = A.outerIndexPtr()[cid + 1];
			int k = 0;
			for (int i = ns; i<ne; i++, k++)
			{
				ridx[k] = A.innerIndexPtr()[i];
				vidx[k] = i;
			}
			return k;
		}

		void CalcJacobiFunc(const Eigen::VectorXf& pTest, SpMat& jac, SpMat& jact)
		{
			ldp::tic();
			CalcJacobiFuncNumeric(pTest, jac, jact);
			//CalcJacobiFuncAnalytic(pTest, jac, jact);
			ldp::toc("jacobi");

#if 1
			// debug
			CalcJacobiFuncNumeric(pTest, jac, jact);
			dumpSparseMatrix(jac_, "D:/num.txt");
			CalcJacobiFuncAnalytic(pTest, jac, jact);
			dumpSparseMatrix(jac_, "D:/ana.txt");
			system("pause");
			// end debug
#endif
		}

		void CalcJacobiFuncAnalytic(const Eigen::VectorXf& pTest, SpMat& jac, SpMat& jact)
		{
			enum { VarPerNode = 6 };
			const int nNodes = x_.size() / 6;
			const int nPixel = imgHeight_ * imgWidth_;

			// data term    ========================================================
//#pragma omp parallel for
			for (int iPixel = 0; iPixel < nPixel; iPixel++)
			{
				int nRow = row_index_of_pixel_[iPixel];
				if (nRow < 0)
					continue;

				int cooPos = coo_pos_of_pixel_[iPixel];

				int corrPixel = map_c2l_corr_[iPixel];
				Tbx::Point3 v(convert(read_float3_from_4(vmap_cano_[iPixel])));
				Tbx::Vec3 n = convert(read_float3_from_4(nmap_cano_[iPixel]));
				Tbx::Point3 vl(convert(read_float3_from_4(vmap_live_[corrPixel])));

				// f
				real f = n.dot(v - vl);

				// partial_psi_partial_f
				real p_psi_p_f = data_term_grad(f);

				KnnIdx knn = vmapKnn_[iPixel];
				Tbx::Dual_quat_cu dq(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
				Tbx::Dual_quat_cu dqk[WarpField::KnnK];
				for (int knnK = 0; knnK < WarpField::KnnK; knnK++)
				{
					int knnNodeId = knn_k(knn, knnK);
					if (knnNodeId < nNodes)
					{
						Tbx::Vec3 r(pTest[knnNodeId * 6], pTest[knnNodeId * 6 + 1], pTest[knnNodeId * 6 + 2]);
						Tbx::Vec3 t(pTest[knnNodeId * 6 + 3], pTest[knnNodeId * 6 + 4], pTest[knnNodeId * 6 + 5]);
						Tbx::Point3 nodesV(convert(read_float3_from_4(nodesVw_[knnNodeId])));
						float invNodesW = nodesVw_[knnNodeId].w;
						dqk[knnK].from_twist(r, t);
						// note: we store inv radius as vw.w, thus using * instead of / here
						dq = dq + dqk[knnK] * exp(-(v - nodesV).dot(v - nodesV)*(2 * invNodesW * invNodesW));
					}// end if (knnNodeId < nNodes)
				}// ebd fir knnK
				Tbx::Dual_quat_cu dq_bar = dq;
				real norm_dq_bar = dq_bar.get_non_dual_part().norm();
				real norm_dq_bar3 = norm_dq_bar*norm_dq_bar*norm_dq_bar;
				dq.normalize();

				// paitial_f_partial_T
				Tbx::Transfo T = Tlw_*dq.to_transformation();
				Tbx::Transfo nvt = outer_product(n, v);
				Tbx::Transfo vlnt = outer_product(n, v);
				Tbx::Transfo p_f_p_T = T*(nvt + nvt.transpose()) - vlnt;

				for (int knnK = 0; knnK < WarpField::KnnK; knnK++)
				{
					int knnNodeId = knn_k(knn, knnK);
					if (knnNodeId < nNodes)
					{
						float p_psi_p_alphak[6];
						Eigen::Triplet<real> coo[6];
						// partial_T_partial_alphak
						for (int ialpha = 0; ialpha < 6; ialpha++)
						{
							Tbx::Transfo p_T_p_alphak = Tbx::Transfo::empty();
							Tbx::Dual_quat_cu p_qk_p_alpha = p_qk_p_alpha_func(dqk[knnK], ialpha);
							for (int idq = 0; idq < 7; idq++)
							{
								// partial_SE3_partial_dqi
								Tbx::Transfo p_SE3_p_dqi = p_SE3_p_dq_func(dq, idq);
								real dq_bar_i = dq_bar[idq];

								// partial_dqi_partial_alphak
								real p_dqi_p_alphak = 0;
								real nodesW = 1 / nodesVw_[knnNodeId].w;
								for (int j = 0; j < 7; j++)
								{
									// partial_dqi_partial_qkj
									real dq_bar_j = dq_bar[j];
									real p_dqi_p_qkj = nodesW / norm_dq_bar;
									if (j <= 3)
										p_dqi_p_qkj -= nodesW / norm_dq_bar3*dq_bar_i*dq_bar_j;

									// partial_qkj_partial_alphak
									real p_qkj_p_alphak = p_qk_p_alpha[j];

									p_dqi_p_alphak += p_dqi_p_qkj * p_qkj_p_alphak;
								}// end for j

								p_T_p_alphak += p_SE3_p_dqi * p_dqi_p_alphak;
							}// end for idq
							p_T_p_alphak = Tlw_ * p_T_p_alphak;

							p_psi_p_alphak[ialpha] = p_psi_p_f * trace_AtB(p_f_p_T, p_T_p_alphak);

							// write to jacobi
							m_cooSys.at(cooPos++) = Eigen::Triplet<real>(nRow,
								knnNodeId * VarPerNode + ialpha, p_psi_p_alphak[ialpha]);

							//if (nRow == 0)
							//{
							//	printf("%d %f\n", nRow, p_f_p_T[0]);
							//	dq_bar.to_transformation().print();
							//	system("pause");
							//}
						}// end for ialpha
					}// end if knnNodeId < nNodes
				}// end for knnK
			}// end for iPixel
			
			// Reg term    ======================================================================
			int nRow = nPixel_rows_;
			int cooPos = nPixel_cooPos_;
			const float lambda = sqrt(param_.fusion_lambda);
			for (int iNode = 0; iNode < nNodes; iNode++)
			{
				KnnIdx knn = nodesKnn_[iNode];
				Tbx::Dual_quat_cu dqi;
				Tbx::Vec3 ri(pTest[iNode * 6], pTest[iNode * 6 + 1], pTest[iNode * 6 + 2]);
				Tbx::Vec3 ti(pTest[iNode * 6 + 3], pTest[iNode * 6 + 4], pTest[iNode * 6 + 5]);
				dqi.from_twist(ri, ti);

				for (int knnK = 0; knnK < WarpField::KnnK; knnK++)
				{
					int knnNodeId = knn_k(knn, knnK);
					if (knnNodeId < nNodes)
					{
						Tbx::Vec3 rj(pTest[knnNodeId * 6], pTest[knnNodeId * 6 + 1], pTest[knnNodeId * 6 + 2]);
						Tbx::Vec3 tj(pTest[knnNodeId * 6 + 3], pTest[knnNodeId * 6 + 4], pTest[knnNodeId * 6 + 5]);
						real alpha_ij = sqrt(max(1 / nodesVw_[iNode].w, 1 / nodesVw_[knnNodeId].w));
						Tbx::Dual_quat_cu dqj;
						dqj.from_twist(rj, tj);
						Tbx::Point3 vj(convert(read_float3_from_4(nodesVw_[knnNodeId])));

						Tbx::Vec3 h = dqi.transform(vj) - dqj.transform(vj);

						// partial_psi_partial_h
						Tbx::Mat3 p_psi_p_h = reg_term_grad(h) * lambda * alpha_ij;

						for (int ialpha = 0; ialpha < VarPerNode; ialpha++)
						{
							Tbx::Transfo p_Ti_p_alpha = p_SE3_p_alpha_func(dqi, ialpha);
							Tbx::Transfo p_Tj_p_alpha = p_SE3_p_alpha_func(dqj, ialpha);

							// partial_h_partial_alpha
							Tbx::Vec3 p_h_p_alphai, p_h_p_alphaj;
							for (int ixyz = 0; ixyz < 3; ixyz++)
							{
								p_h_p_alphai[ixyz] = p_Ti_p_alpha(ixyz, 0) * vj.x + p_Ti_p_alpha(ixyz, 1) * vj.y
									+ p_Ti_p_alpha(ixyz, 2) * vj.z + p_Ti_p_alpha(ixyz, 3);
								p_h_p_alphaj[ixyz] = -(p_Tj_p_alpha(ixyz, 0) * vj.x + p_Tj_p_alpha(ixyz, 1) * vj.y
									+ p_Tj_p_alpha(ixyz, 2) * vj.z + p_Tj_p_alpha(ixyz, 3));
							}

							// partial_psi_partial_alpha
							Tbx::Vec3 p_psi_p_alphai = p_psi_p_h * p_h_p_alphai;
							Tbx::Vec3 p_psi_p_alphaj = p_psi_p_h * p_h_p_alphaj;
							for (int ixyz = 0; ixyz < 3; ixyz++)
							{
								m_cooSys[cooPos++] = Eigen::Triplet<real>(nRow + ixyz,
									iNode * VarPerNode + ialpha, 
									p_psi_p_alphai[ixyz]);
								m_cooSys[cooPos++] = Eigen::Triplet<real>(nRow + ixyz,
									knnNodeId * VarPerNode + ialpha,
									p_psi_p_alphaj[ixyz]);
							}
						}// end for ialpha
						nRow += 3;
					}// end if knnNode valid
				}// end for knnK
			}// end for iNode
	
			if (m_cooSys.size())
				jac.setFromTriplets(m_cooSys.begin(), m_cooSys.end());
			jact = jac.transpose();
		}

		void CalcJacobiFuncNumeric(const Eigen::VectorXf& pTest, SpMat& jac, SpMat& jact)
		{
			Vec p = pTest;
			const int nobs = jac.rows();
			const int nvars = jac.cols();

			register int i, j, jj, k;
			register real d;
			int ii, m, *jcol, *varlist, *coldone, forw;
			int *vidxs, *ridxs;
			real *tmpd;
			real delta, delta_mul;

			/* retrieve problem-specific information passed in *dat */
			Vec hx(nobs), hxx(nobs);
			forw = 1;
			delta_mul = real(1e-3);
			delta = real(1e-5);

			CalcEnergyFunc(p, hx);//hx = f(p)

			jcol = (int *)malloc(nobs*sizeof(int)); /* keeps track of measurements influenced by the set of variables currently in "varlist" below */
			for (i = 0; i<nobs; ++i) jcol[i] = -1;

			vidxs = (int *)malloc(2 * nobs*sizeof(int));
			ridxs = vidxs + nobs;

			varlist = (int *)malloc(nvars*sizeof(int)); /* stores indices of J's columns which are computed with the same "func" call */
			coldone = (int *)malloc(nvars*sizeof(int)); /* keeps track of J's columns which have been already computed */
			memset(coldone, 0, nvars*sizeof(int)); /* initialize to zero */

			tmpd = (real *)malloc(nvars*sizeof(real));

			for (j = 0; j<nvars; ++j)
			{
				real scl;

				if (coldone[j]) continue; /* column j already computed */

				//for(i=0; i<nobs; ++i) jcol[i]=-1;
				k = FindColIndices(jac, j, vidxs, ridxs);
				for (i = 0; i<k; ++i) jcol[ridxs[i]] = j;
				varlist[0] = j; m = 1; coldone[j] = 1;

				for (jj = j + 1; jj<nvars; ++jj)
				{
					if (coldone[jj]) continue; /* column jj already computed */

					k = FindColIndices(jac, jj, vidxs, ridxs);
					for (i = 0; i<k; ++i)
					if (jcol[ridxs[i]] != -1) goto nextjj;

					if (k == 0) { coldone[jj] = 1; continue; } /* all zeros column, ignore */

					/* column jj does not clash with previously considered ones, mark it */
					for (i = 0; i<k; ++i) jcol[ridxs[i]] = jj;
					varlist[m++] = jj; coldone[jj] = 1;

				nextjj:
					continue;
				}

				for (k = 0; k<m; ++k)
				{
					/* determine d=max(SPLM_DELTA_SCALE*|p[varlist[k]]|, delta), see HZ */
					d = delta_mul*p[varlist[k]]; // force evaluation
					d = fabs(d);
					if (d<delta) d = delta;

					tmpd[varlist[k]] = d;
					p[varlist[k]] += d;
				}

				CalcEnergyFunc(p, hxx);// hxx=f(p+d)

				if (forw)
				{
					for (k = 0; k<m; ++k)
						p[varlist[k]] -= tmpd[varlist[k]]; /* restore */

					scl = 1.0;
				}
				else
				{ // central
					for (k = 0; k<m; ++k)
						p[varlist[k]] -= 2 * tmpd[varlist[k]];

					CalcEnergyFunc(p, hx);// hx=f(p-d)

					for (k = 0; k<m; ++k)
						p[varlist[k]] += tmpd[varlist[k]]; /* restore */

					scl = 0.5; // 1./2.
				}

				for (k = 0; k<m; ++k)
				{
					d = tmpd[varlist[k]];
					d = scl / d; /* invert so that divisions can be carried out faster as multiplications */

					jj = FindColIndices(jac, varlist[k], vidxs, ridxs);
					for (i = 0; i<jj; ++i)
					{
						ii = ridxs[i];
						jac.valuePtr()[vidxs[i]] = (hxx[ii] - hx[ii])*d;
						jcol[ii] = -1; /* restore */
					}
				}
			}//end for jj

			//calc 
			jact = jac.transpose();

			free(tmpd);
			free(coldone);
			free(varlist);
			free(vidxs);
			free(jcol);
		}

		static void dumpSparseMatrix(const SpMat& A, const char* filename)
		{
			FILE* pFile = fopen(filename, "w");
			if (!pFile)
				throw std::exception("dumpSparseMatrix: create file failed!");
			for (int r = 0; r < A.outerSize(); r++)
			{
				int rs = A.outerIndexPtr()[r];
				int re = A.outerIndexPtr()[r+1];
				for (int c = rs; c < re; c++)
					fprintf(pFile, "%d %d %f\n", r, A.innerIndexPtr()[c], A.valuePtr()[c]);
			}
			fclose(pFile);
		}
	};

	CpuGaussNewton::CpuGaussNewton()
	{
		m_egc = new EigenContainter();
	}

	CpuGaussNewton::~CpuGaussNewton()
	{
		delete m_egc;
	}

	void CpuGaussNewton::init(WarpField* pWarpField, const MapArr& vmap_cano, const MapArr& nmap_cano,
		Param param, Intr intr)
	{
		m_pWarpField = pWarpField;
		m_egc->imgWidth_ = vmap_cano.cols();
		m_egc->imgHeight_ = vmap_cano.rows();
		m_egc->param_ = param;
		m_egc->intr_ = intr;
		m_egc->Tlw_ = pWarpField->get_rigidTransform();

		// copy maps to cpu
		m_egc->vmap_cano_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		m_egc->nmap_cano_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		vmap_cano.download(m_egc->vmap_cano_.data(), m_egc->imgWidth_*sizeof(float4));
		nmap_cano.download(m_egc->nmap_cano_.data(), m_egc->imgWidth_*sizeof(float4));

		// extract knn map and copy to cpu
		m_pWarpField->extract_knn_for_vmap(vmap_cano, m_vmapKnn);
		m_egc->vmapKnn_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		m_vmapKnn.download(m_egc->vmapKnn_.data(), m_egc->imgWidth_*sizeof(WarpField::KnnIdx));

		//extract nodes info and copy to cpu
		m_pWarpField->extract_nodes_info(m_nodesKnn, m_twist, m_vw);
		m_egc->nodesKnn_.resize(m_nodesKnn.size());
		m_nodesKnn.download(m_egc->nodesKnn_.data());
		m_egc->x_.resize(m_twist.size());
		m_twist.download(m_egc->x_.data());
		m_egc->nodesVw_.resize(m_vw.size());
		m_vw.download(m_egc->nodesVw_.data());
	}

	void CpuGaussNewton::findCorr(const MapArr& vmap_live, const MapArr& nmap_live,
		const MapArr& vmap_warp, const MapArr& nmap_warp)
	{
		m_egc->vmap_live_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		m_egc->nmap_live_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		vmap_live.download(m_egc->vmap_live_.data(), m_egc->imgWidth_*sizeof(float4));
		nmap_live.download(m_egc->nmap_live_.data(), m_egc->imgWidth_*sizeof(float4));
		m_egc->vmap_warp_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		m_egc->nmap_warp_.resize(m_egc->imgHeight_*m_egc->imgWidth_);
		vmap_warp.download(m_egc->vmap_warp_.data(), m_egc->imgWidth_*sizeof(float4));
		nmap_warp.download(m_egc->nmap_warp_.data(), m_egc->imgWidth_*sizeof(float4));
		m_egc->CalcCorr();
	}

	void CpuGaussNewton::solve()
	{
		m_egc->Optimize(m_egc->x_, m_egc->param_.fusion_GaussNewton_maxIter);

		m_twist.upload(m_egc->x_.data(), m_egc->x_.size());
		m_pWarpField->update_nodes_via_twist(m_twist);
	}
}