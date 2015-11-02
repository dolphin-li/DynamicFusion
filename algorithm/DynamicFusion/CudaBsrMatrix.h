#pragma once

#include <device_array.h>
#include <cusparse.h>
#include <channel_descriptor.h>
#include <texture_types.h>
#include <texture_fetch_functions.h>
#include "cuda_utils.h"

class CudaBsrMatrix
{
public:
	CudaBsrMatrix(cusparseHandle_t handle);
	~CudaBsrMatrix();

	void clear();
	void resize(int blocksInRow, int blocksInCol, int rowsPerBlock, int colsPerBlock);
	void resize(int blocksInRow, int blocksInCol, int sizePerBlock);

	int blocksInRow()const{ return m_blocksInRow; }
	int blocksInCol()const{ return m_blocksInCol; }
	int rowsPerBlock() const{ return m_rowsPerBlock; }
	int colsPerBlock() const{ return m_colsPerBlock; }
	int rows()const{ return m_blocksInRow * m_rowsPerBlock; }
	int cols()const{ return m_blocksInCol * m_colsPerBlock; }
	int nnz()const{ return m_nnz; }
	bool isSquareBlock()const{ return m_rowsPerBlock == m_colsPerBlock; }
	const float* value()const{ return m_values.ptr(); }
	float* value(){ return m_values.ptr(); }
	const int* bsrRowPtr()const{ return m_bsrRowPtr.ptr(); }
	int* bsrRowPtr(){ return m_bsrRowPtr.ptr(); }
	const int* bsrRowPtr_coo()const{ return m_bsrRowPtr_coo.ptr(); }
	int* bsrRowPtr_coo(){ return m_bsrRowPtr_coo.ptr(); }
	const int* bsrColIdx()const{ return m_bsrColIdx.ptr(); }
	int* bsrColIdx(){ return m_bsrColIdx.ptr(); }

	CudaBsrMatrix& operator = (float constVal);
	CudaBsrMatrix& operator = (const CudaBsrMatrix& rhs);
	void setValue(const float* val_d);

	// return alpha * this + beta
	CudaBsrMatrix& axpy(float alpha, float beta = 0.f);

private:
	cusparseHandle_t m_cusparseHandle;
	int m_blocksInRow;
	int m_blocksInCol;
	int m_rowsPerBlock;
	int m_colsPerBlock;
	int m_nnz;
	cudaTextureObject_t m_tex_values;
	cudaTextureObject_t m_tex_bsrRowPtr;
	cudaTextureObject_t m_tex_bsrRowPtr_coo;
	cudaTextureObject_t m_tex_bsrColIdx;

	DeviceArray<int> m_bsrRowPtr;
	DeviceArray<int> m_bsrRowPtr_coo;
	DeviceArray<int> m_bsrColIdx;
	DeviceArray<float> m_values;
};
