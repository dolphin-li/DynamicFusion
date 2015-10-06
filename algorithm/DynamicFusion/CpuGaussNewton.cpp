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

				// debug
				//dumpSparseMatrix(jac_, "D:/1.txt");
				//system("pause");
				// end debug

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
			return v;
		}
		inline real reg_term_penalty(real v)const
		{
			return v;
		}
		inline Tbx::Transfo outter_product(Tbx::Vec3 n, Tbx::Point3 v)const
		{
			return Tbx::Transfo(
				n.x*v.x, n.x*v.y, n.x*v.z, n.x,
				n.y*v.x, n.y*v.y, n.y*v.z, n.y,
				n.z*v.x, n.z*v.y, n.z*v.z, n.z,
				0, 0, 0, 0
				);
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

			// data term
			int nRow = 0;
			for (int iPixel = 0; iPixel < nPixel; iPixel++)
			{
				int corrPixel = map_c2l_corr_[iPixel];
				if (corrPixel < 0)
					continue;

				bool valid = false;
				KnnIdx knn = vmapKnn_[iPixel];
				for (int k = 0; k < WarpField::KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						valid = true;
						for (int t = 0; t < VarPerNode; t++)
							m_cooSys.push_back(Eigen::Triplet<real>(nRow, 
							knnNodeId * VarPerNode + t, 0));
					}
				}
				if (valid)
					nRow++;
			}// end for iPixel

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
			int nRow = 0;
			for (int iPixel = 0; iPixel < map_c2l_corr_.size(); iPixel++)
			{
				int corrPixel = map_c2l_corr_[iPixel];
				if (corrPixel < 0)
					continue;

				Tbx::Vec3 v = convert(read_float3_from_4(vmap_cano_[iPixel]));
				Tbx::Vec3 n = convert(read_float3_from_4(nmap_cano_[iPixel]));
				Tbx::Vec3 vl = convert(read_float3_from_4(vmap_live_[corrPixel]));

				bool valid = false;
				KnnIdx knn = vmapKnn_[iPixel];
				Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
				for (int k = 0; k < WarpField::KnnK; k++)
				{
					int knnNodeId = knn_k(knn, k);
					if (knnNodeId < nNodes)
					{
						valid = true;
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
				if (valid)
				{
					dq_blend.normalize();
					v = Tlw_*(dq_blend.transform(Tbx::Point3(v)));
					n = Tlw_*(dq_blend.rotate(n));
					f[nRow] = data_term_penalty(n.dot(v - vl));
					nRow++;
				}
			}// end for iPixel

			// reg term
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
						f[nRow++] = reg_term_penalty(val.x) * lambda * alpha_ij;
						f[nRow++] = reg_term_penalty(val.y) * lambda * alpha_ij;
						f[nRow++] = reg_term_penalty(val.z) * lambda * alpha_ij;
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
			CalcJacobiFuncNumeric(pTest, jac, jact);
		}

		void CalcJacobiFuncAnalytic(const Eigen::VectorXf& pTest, SpMat& jac, SpMat& jact)
		{
			enum { VarPerNode = 6 };
			const int nNodes = x_.size() / 6;
			const int nPixel = imgHeight_ * imgWidth_;

			// data term
			int nRow = 0;
			for (int iPixel = 0; iPixel < nPixel; iPixel++)
			{
				int corrPixel = map_c2l_corr_[iPixel];
				if (corrPixel < 0)
					continue;

				Tbx::Vec3 v = convert(read_float3_from_4(vmap_cano_[iPixel]));
				Tbx::Vec3 n = convert(read_float3_from_4(nmap_cano_[iPixel]));
				Tbx::Vec3 vl = convert(read_float3_from_4(vmap_live_[corrPixel]));
				real f = n.dot(v - vl);
				real p_psi_p_f = f; // corr. to ||.||^2 penalty

				bool valid = false;
				KnnIdx knn = vmapKnn_[iPixel];
				Tbx::Dual_quat_cu dq_blend(Tbx::Quat_cu(0, 0, 0, 0), Tbx::Quat_cu(0, 0, 0, 0));
				for (int knnK = 0; knnK < WarpField::KnnK; knnK++)
				{
					int knnNodeId = knn_k(knn, knnK);
					if (knnNodeId < nNodes)
					{
						valid = true;
						Tbx::Dual_quat_cu dq;
						Tbx::Vec3 r(pTest[knnNodeId * 6], pTest[knnNodeId * 6 + 1], pTest[knnNodeId * 6 + 2]);
						Tbx::Vec3 t(pTest[knnNodeId * 6 + 3], pTest[knnNodeId * 6 + 4], pTest[knnNodeId * 6 + 5]);
						Tbx::Vec3 nodesV = convert(read_float3_from_4(nodesVw_[knnNodeId]));
						float nodesW = nodesVw_[knnNodeId].w;
						dq.from_twist(r, t);
						// note: we store inv radius as vw.w, thus using * instead of / here
						dq_blend = dq_blend + dq*exp(-(v - nodesV).dot(v - nodesV)*(2 * nodesW*nodesW));
					}// end if (knnNodeId < nNodes)
				}// ebd fir knnK

				Tbx::Transfo T = Tlw_*dq_blend.to_transformation();
				

				// to the next row
				if (valid)
					nRow++;
			}// end for iPixel
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
			real delta;

			/* retrieve problem-specific information passed in *dat */
			Vec hx(nobs), hxx(nobs);
			forw = 1;
			delta = real(1e-6);

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
					d = real(1e-3)*p[varlist[k]]; // force evaluation
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