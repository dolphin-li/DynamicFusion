#pragma once

#include <device_array.h>
#include <cusparse.h>
#include <channel_descriptor.h>
#include <texture_types.h>
#include <texture_fetch_functions.h>
#include "cuda_utils.h"
#include <algorithm>
class CudaDiagBlockMatrix;
class CudaBsrMatrix
{
public:
	enum{
		CTA_SIZE = 512
	};
	struct Range
	{
		CudaBsrMatrix* A;
		int blockRowBegin;
		int blockRowEnd;
		int blockColBegin;
		int blockColEnd;
		Range(const CudaBsrMatrix* A_, int blockRowBegin_ = 0, int blockColBegin_ = 0,
			int blockRowEnd_=INT_MAX, int blockColEnd_=INT_MAX)
		{
			if (A_ == nullptr)
				throw std::exception("CudaBsrMatrix::Range(): nullptr error!");
			A = (CudaBsrMatrix*)A_;
			blockRowBegin = std::max(0, blockRowBegin_);
			blockRowEnd = std::min(A->blocksInRow(), blockRowEnd_);
			blockColBegin = std::max(0, blockColBegin_);
			blockColEnd = std::min(A->blocksInCol(), blockColEnd_);
		}

		int blocksInRow()const{ return blockRowEnd - blockRowBegin; }
		int blocksInCol()const{ return blockColEnd - blockColBegin; }
		int rowsPerBlock() const{ return A->rowsPerBlock(); }
		int colsPerBlock() const{ return A->colsPerBlock(); }
		int rows()const{ return blocksInRow() * rowsPerBlock(); }
		int cols()const{ return blocksInCol() * colsPerBlock(); }

		// C = alpha * this * B; Assume structure is given
		void multBsr_structure(const CudaBsrMatrix& B, CudaBsrMatrix& C, const CudaBsrMatrix* D = nullptr)const;

		// C = alpha * this * B + beta * D; Assume structure is given
		void multBsr_value(const Range& B, CudaBsrMatrix& C, float alpha = 1.f,
			const Range* D = nullptr, float beta = 0.f)const;

		// C = alpha * this * B' + beta * D; Assume structure is given
		void multBsrT_value(const Range& B, CudaBsrMatrix& C, float alpha = 1.f,
			const Range* D = nullptr, float beta = 0.f)const;
		void multBsrT_addDiag_value(const Range& B, CudaBsrMatrix& C, float alpha = 1.f,
			const CudaDiagBlockMatrix* D = nullptr, float beta = 0.f)const;

		// compute C = alpha * blockDiag(this*this') + beta*C;
		// if lowerInsteadOfFull, then only the lower triangular part is touched
		void AAt_blockDiags(CudaDiagBlockMatrix& C, bool lowerInsteadOfFull,
			float alpha = 1.f, float beta = 0.f)const;
	};
public:
	CudaBsrMatrix(cusparseHandle_t handle, bool symbolic=false);
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
	void toCsr(DeviceArray<int>& csrRowPtr, DeviceArray<int>& csrColIdx, DeviceArray<float>& csrValue)const;

	// B will be with block size 1x1, thus csr
	void toCsr_structure(CudaBsrMatrix& B)const;
	void toCsr_value(CudaBsrMatrix& B)const;

	// sub matrix by rows
	void subRows_structure(CudaBsrMatrix& S, int blockRowBegin, int blockRowEnd)const;
	void subRows_value(CudaBsrMatrix& S, int blockRowBegin, int blockRowEnd)const;

	// copy the row ptr to this, begin/endConstructRowPtr called inside
	void setRowFromBsrRowPtr(const int* bsrRowPtr);

	// convert from blocked csrRowPtr to this, begin/endConstructRowPtr called inside
	// blocked csrRowPtr indicates that the input is blocked: its value is inherently block-by-block
	void setRowFromBlockedCsrRowPtr(const int* csrRowPtr);

	/// resize nnz related memories, called inside endConstructRowPtr() functions
	// generally you do not need to call this function explicitly
	void resize_nnzBlocks(int nnzBlocks);

	//
	bool isSymbolic()const{ return m_symbolic; }
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
	Range range(int blockRowBegin_ = 0, int blockColBegin_ = 0,
		int blockRowEnd_ = INT_MAX, int blockColEnd_ = INT_MAX)const{
		return Range(this, blockRowBegin_, blockColBegin_, blockRowEnd_, blockColEnd_);
	}

	CudaBsrMatrix& operator = (float constVal);
	CudaBsrMatrix& operator = (const CudaBsrMatrix& rhs);
	void setValue(const float* val_d);

	// return alpha * this + beta
	CudaBsrMatrix& axpy(float alpha, float beta = 0.f);

	// this(i,i) = alpha * this(i,i) + beta
	CudaBsrMatrix& axpy_diag(float alpha, float beta = 0.f);

	// mult-vector: y = alpha * this * x + beta * y
	void Mv(const float* x, float* y, float alpha = 1.f, float beta = 0.f)const;

	// C = alpha*this*B + beta*D (if D is not null)
	void multBsr_structure(const CudaBsrMatrix& B, CudaBsrMatrix& C, const CudaBsrMatrix* D=nullptr)const;

	// C = alpha*this*B + beta*D
	void multBsr_value(const CudaBsrMatrix& B, CudaBsrMatrix& C, float alpha = 1.f,
		const CudaBsrMatrix* D = nullptr, float beta = 0.f)const;

	// C = alpha*this*B' + beta*D
	void multBsrT_value(const CudaBsrMatrix& B, CudaBsrMatrix& C, float alpha = 1.f,
		const CudaBsrMatrix* D = nullptr, float beta = 0.f)const;
	void multBsrT_addDiag_value(const CudaBsrMatrix& B, CudaBsrMatrix& C, float alpha = 1.f,
		const CudaDiagBlockMatrix* D = nullptr, float beta = 0.f)const;

	// compute C = alpha * blockDiag(this*this') + beta*C;
	// if lowerInsteadOfFull, then only the lower triangular part is touched
	void AAt_blockDiags(CudaDiagBlockMatrix& C, bool lowerInsteadOfFull,
		float alpha = 1.f, float beta = 0.f)const;

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
	char* get_helper_buffer(int nBytes)const;
private:
	cusparseHandle_t m_cusparseHandle;
	cusparseMatDescr_t m_desc;
	csrgemm2Info_t m_csrgemm2info;
	int m_blocksInRow;
	int m_blocksInCol;
	int m_rowsPerBlock;
	int m_colsPerBlock;
	int m_nnzBlocks;
	bool m_symbolic;
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
