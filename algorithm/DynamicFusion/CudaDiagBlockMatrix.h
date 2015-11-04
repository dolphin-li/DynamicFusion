#pragma once

#include <device_array.h>
#include <channel_descriptor.h>
#include <texture_types.h>
#include <texture_fetch_functions.h>
#include "cuda_utils.h"

class CudaDiagBlockMatrix
{
	enum{
		CTA_SIZE = 512
	};
	struct Range
	{
		CudaDiagBlockMatrix* A;
		int blockBegin;
		int blockEnd;
		Range(CudaDiagBlockMatrix* A_, int blockBegin_ = 0,
			int blockEnd_ = INT_MAX)
		{
			A = A_;
			blockBegin = blockBegin_;
			blockEnd = blockEnd_;
		}
	};
public:
	CudaDiagBlockMatrix();
	~CudaDiagBlockMatrix();

	void clear();
	void resize(int numBlocks, int blockSize);

	int blockSize()const{ return m_blockSize; }
	int numBlocks()const{ return m_numBlocks; }
	int rows()const{ return m_numBlocks * m_blockSize; }
	int cols()const{ return m_numBlocks * m_blockSize; }
	int nnz()const{ return m_blockSize*m_blockSize*m_numBlocks; }
	const float* value()const{ return m_values.ptr(); }
	float* value(){ return m_values.ptr(); }
	cudaTextureObject_t getTexture()const{ return m_tex; }

	CudaDiagBlockMatrix& operator = (float constVal);
	CudaDiagBlockMatrix& operator = (const CudaDiagBlockMatrix& rhs);
	DeviceArray<float> toDeviceArray(){ return DeviceArray<float>(value(), nnz()); }
	void setValue(const float* val_d);

	// return alpha * this + beta
	CudaDiagBlockMatrix& axpy(float alpha, float beta = 0.f);

	// this(i,i) = alpha * this(i,i) + beta
	CudaDiagBlockMatrix& axpy_diag(float alpha, float beta = 0.f);

	// cholesky decomposition, results overwrite 
	// the lower part of the row majored matrix.
	// assume positive definate
	CudaDiagBlockMatrix& cholesky();

	// inverse the lower triangular part of the matrix
	CudaDiagBlockMatrix& invL();

	// Upper(this) = Lower(this)^t
	CudaDiagBlockMatrix& transpose_L_to_U();

	// compute result = L'*L, where L is the lower triangular part of this
	void LtL(CudaDiagBlockMatrix& result)const;

	// vec_out = alpha * Lower(this) * vec_in + beta;
	void Lv(const float* vec_in, float* vec_out, float alpha = 1.f, float beta = 0.f);

	// vec_out = alpha * Lower(this)^t * vec_in + beta;
	void Ltv(const float* vec_in, float* vec_out, float alpha = 1.f, float beta = 0.f);

	Range range(int blockBegin_ = 0, int blockEnd_ = INT_MAX){
		return Range(this, blockBegin_, blockEnd_);
	}
private:
	int m_blockSize;
	int m_numBlocks;
	cudaTextureObject_t m_tex;

	DeviceArray<float> m_values;
};

