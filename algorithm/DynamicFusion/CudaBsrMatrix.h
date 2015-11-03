#pragma once

#include <device_array.h>
#include <cusparse.h>
#include <channel_descriptor.h>
#include <texture_types.h>
#include <texture_fetch_functions.h>
#include "cuda_utils.h"
class CudaDiagBlockMatrix;
class CudaBsrMatrix
{
	enum{
		CTA_SIZE = 512
	};
public:
	CudaBsrMatrix(cusparseHandle_t handle);
	~CudaBsrMatrix();

	void clear();

	//
	void resize(int blocksInRow, int blocksInCol, int rowsPerBlock, int colsPerBlock);
	void resize(int blocksInRow, int blocksInCol, int sizePerBlock);

	//// to construct a bsr matrix
	// 0. call resize
	// 1. call beginConstructRowPtr(), fill in the bsrRowPtr()[0:rows()]; then call endConstructRowPtr()
	// 2. fill in the bsrColIdx()[0:nnz()-1];
	// 3. once structure defined, you can fill values by free into value()
	void beginConstructRowPtr();
	void endConstructRowPtr();
	// similar with endConstructRowPtr, but nnzBlocks in supplied by host side
	void endConstructRowPtr(int nnzBlocks);

	// after calling resize, you may also call this function to convert a csr matrix to this
	void fromCsr(const int* csrRowPtr, const int* csrColIdx, const float* csrValue);
	void toCsr(DeviceArray<int>& csrRowPtr, DeviceArray<int>& csrColIdx, DeviceArray<float>& csrValue);

	// copy the row ptr to this, begin/endConstructRowPtr called inside
	void setRowFromBsrRowPtr(const int* bsrRowPtr);

	// convert from blocked csrRowPtr to this, begin/endConstructRowPtr called inside
	// blocked csrRowPtr indicates that the input is blocked: its value is inherently block-by-block
	void setRowFromBlockedCsrRowPtr(const int* csrRowPtr);

	/// resize nnz related memories, called inside endConstructRowPtr() functions
	// generally you do not need to call this function explicitly
	void resize_nnzBlocks(int nnzBlocks);

	//
	int blocksInRow()const{ return m_blocksInRow; }
	int blocksInCol()const{ return m_blocksInCol; }
	int rowsPerBlock() const{ return m_rowsPerBlock; }
	int colsPerBlock() const{ return m_colsPerBlock; }
	int rows()const{ return m_blocksInRow * m_rowsPerBlock; }
	int cols()const{ return m_blocksInCol * m_colsPerBlock; }
	int nnz()const{ return m_nnzBlocks * m_rowsPerBlock * m_colsPerBlock; }
	int nnzBlocks()const{ return m_nnzBlocks; }
	bool isSquareBlock()const{ return m_rowsPerBlock == m_colsPerBlock; }
	const float* value()const{ return m_values.ptr(); }
	float* value(){ return m_values.ptr(); }
	cudaTextureObject_t valueTexture()const{ return m_tex_values; }
	const int* bsrRowPtr()const{ return m_bsrRowPtr.ptr(); }
	int* bsrRowPtr(){ return m_bsrRowPtr.ptr(); }
	cudaTextureObject_t bsrRowPtrTexture()const{ return m_tex_bsrRowPtr; }
	const int* bsrRowPtr_coo()const{ return m_bsrRowPtr_coo.ptr(); }
	int* bsrRowPtr_coo(){ return m_bsrRowPtr_coo.ptr(); }
	cudaTextureObject_t bsrRowPtr_cooTexture()const{ return m_tex_bsrRowPtr_coo; }
	const int* bsrColIdx()const{ return m_bsrColIdx.ptr(); }
	int* bsrColIdx(){ return m_bsrColIdx.ptr(); }
	cudaTextureObject_t bsrColIdxTexture()const{ return m_tex_bsrColIdx; }
	cusparseMatDescr_t getCuSparseMatDesc()const{ return m_desc; }

	CudaBsrMatrix& operator = (float constVal);
	CudaBsrMatrix& operator = (const CudaBsrMatrix& rhs);
	void setValue(const float* val_d);

	// return alpha * this + beta
	CudaBsrMatrix& axpy(float alpha, float beta = 0.f);

	// mult-vector: y = alpha * this * x + beta * y
	void Mv(const float* x, float* y, float alpha = 1.f, float beta = 0.f)const;

	// mult-matrix: y = alpha * this * x + beta * y
	// if useLowerInsteadOfFull, then only the lower triangular part will be considered
	// if trans, then x is implicitly transposed and then mult.
	void rightMultDiag_structure(const CudaDiagBlockMatrix& x, CudaBsrMatrix& y)const;
	void rightMultDiag_value(const CudaDiagBlockMatrix& x, CudaBsrMatrix& y, 
		bool useLowerInsteadOfFull_x, bool trans_x, float alpha = 1.f, float beta=0.f)const;

	// separate transpose into two phases: structure and value
	void transposeStructureTo(CudaBsrMatrix& rhs)const;
	void transposeValueTo(CudaBsrMatrix& rhs)const;

	void dump(std::string name)const;
protected:
	static void fill_increment_1_n(int* data, int n);
	void transpose_fill_values_by_blockId(const int* blockIds, const CudaBsrMatrix& t);
private:
	cusparseHandle_t m_cusparseHandle;
	cusparseMatDescr_t m_desc;
	int m_blocksInRow;
	int m_blocksInCol;
	int m_rowsPerBlock;
	int m_colsPerBlock;
	int m_nnzBlocks;
	cudaTextureObject_t m_tex_values;
	cudaTextureObject_t m_tex_bsrRowPtr;
	cudaTextureObject_t m_tex_bsrRowPtr_coo;
	cudaTextureObject_t m_tex_bsrColIdx;

	DeviceArray<int> m_bsrRowPtr;
	DeviceArray<int> m_bsrRowPtr_coo;
	DeviceArray<int> m_bsrColIdx;
	mutable DeviceArray<int> m_helperBuffer;
	DeviceArray<float> m_values;
};
