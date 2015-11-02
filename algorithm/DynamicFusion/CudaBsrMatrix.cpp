#include "CudaBsrMatrix.h"


CudaBsrMatrix::CudaBsrMatrix(cusparseHandle_t handle)
{
	m_cusparseHandle = handle;
	m_blocksInRow = 0;
	m_blocksInCol = 0;
	m_rowsPerBlock = 0;
	m_colsPerBlock = 0;

	m_tex_values = 0;
	m_tex_bsrRowPtr = 0;
	m_tex_bsrRowPtr_coo = 0;
	m_tex_bsrColIdx = 0;
}

CudaBsrMatrix::~CudaBsrMatrix()
{
	clear();
}

void CudaBsrMatrix::clear()
{
	m_cusparseHandle = nullptr;
	m_blocksInRow = 0;
	m_blocksInCol = 0;
	m_rowsPerBlock = 0;
	m_colsPerBlock = 0;

	if (m_tex_values)
		cudaSafeCall(cudaDestroyTextureObject(m_tex_values), "CudaBsrMatrix::clear(), destroy tex 1");
	if (m_tex_bsrRowPtr)
		cudaSafeCall(cudaDestroyTextureObject(m_tex_bsrRowPtr), "CudaBsrMatrix::clear(), destroy tex 2");
	if (m_tex_bsrRowPtr_coo)
		cudaSafeCall(cudaDestroyTextureObject(m_tex_bsrRowPtr_coo), "CudaBsrMatrix::clear(), destroy tex 3");
	if (m_tex_bsrColIdx)
		cudaSafeCall(cudaDestroyTextureObject(m_tex_bsrColIdx), "CudaBsrMatrix::clear(), destroy tex 4");

	m_tex_values = 0;
	m_tex_bsrRowPtr = 0;
	m_tex_bsrRowPtr_coo = 0;
	m_tex_bsrColIdx = 0;
}

void CudaBsrMatrix::resize(int blocksInRow, int blocksInCol, int rowsPerBlock, int colsPerBlock)
{
	m_blocksInRow = blocksInRow;
	m_blocksInCol = blocksInCol;
}

void CudaBsrMatrix::resize(int blocksInRow, int blocksInCol, int sizePerBlock)
{
	resize(blocksInRow, blocksInCol, sizePerBlock, sizePerBlock);
}
