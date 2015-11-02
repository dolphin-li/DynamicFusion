#include "CudaDiagBlockMatrix.h"
#include "GpuCholeSky.h"

CudaDiagBlockMatrix::CudaDiagBlockMatrix()
{
	m_blockSize = 0;
	m_numBlocks = 0;
	m_tex = 0;
}

CudaDiagBlockMatrix::~CudaDiagBlockMatrix()
{
	clear();
}

void CudaDiagBlockMatrix::clear()
{
	m_blockSize = 0;
	m_numBlocks = 0;
	if (m_tex)
		cudaSafeCall(cudaDestroyTextureObject(m_tex), "CudaDiagBlockMatrix::clear(), destroy tex");
	m_tex = 0;
	m_values.release();
}

void CudaDiagBlockMatrix::resize(int numBlocks, int blockSize)
{
	m_numBlocks = numBlocks;
	m_blockSize = blockSize;
	if (nnz() > m_values.size())
	{
		m_values.create(nnz() * 1.2);

		// bind to texture
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeLinear;
		texRes.res.linear.devPtr = value();
		texRes.res.linear.sizeInBytes = m_values.sizeBytes();
		texRes.res.linear.desc = cudaCreateChannelDesc<float>();
		cudaTextureDesc texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = 0;
		texDescr.filterMode = cudaFilterModePoint;
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeElementType;
		cudaSafeCall(cudaCreateTextureObject(&m_tex, &texRes, &texDescr, NULL),
			"CudaDiagBlockMatrix::bindTexture");
	}
}

CudaDiagBlockMatrix& CudaDiagBlockMatrix::cholesky()
{
	gpu_cholesky::single_thread_cholesky_batched(value(), blockSize(),
		blockSize()*blockSize(), numBlocks());
	return *this;
}

CudaDiagBlockMatrix& CudaDiagBlockMatrix::invL()
{
	gpu_cholesky::single_thread_tril_inv_batched(value(), blockSize(),
		blockSize()*blockSize(), numBlocks());
	return *this;
}

void CudaDiagBlockMatrix::LtL(CudaDiagBlockMatrix& result)const
{
	result.resize(numBlocks(), blockSize());
	gpu_cholesky::single_thread_LtL_batched(
		result.value(), blockSize()*blockSize(), value(),
		blockSize()*blockSize(), blockSize(), numBlocks());
}

CudaDiagBlockMatrix& CudaDiagBlockMatrix::operator = (const CudaDiagBlockMatrix& rhs)
{
	resize(rhs.numBlocks(), rhs.blockSize());
	cudaSafeCall(cudaMemcpy(value(), rhs.value(), nnz()*sizeof(float),
		cudaMemcpyDeviceToDevice), "CudaDiagBlockMatrix::operator = rhs");
	return *this;
}

void CudaDiagBlockMatrix::setValue(const float* val_d)
{
	cudaSafeCall(cudaMemcpy(value(), val_d, nnz()*sizeof(float),
		cudaMemcpyDeviceToDevice), "CudaDiagBlockMatrix::setValue");
}

