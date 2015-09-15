
#include "ldp_basic_vec.h"
#include "CFreeImage.h"
#include "FreeImage.h"
#pragma comment(lib, "FreeImage.lib")
using namespace std;

CFreeImage::CFreeImage()
{
	clear();
}

CFreeImage::CFreeImage(const CFreeImage& other)
{
	_width = other._width;
	_height = other._height;
	_nChannels = other._nChannels;
	_bits.assign(other._bits.begin(), other._bits.end());
}

CFreeImage::~CFreeImage()
{
	clear();
}

void CFreeImage::clear()
{
	_width = _height = _nChannels = 0;
	_bits.clear();
}

void CFreeImage::rgb2bgr()
{
	if(_nChannels < 3)
		return;
	if(_bits.size() == 0)
		return;
	int size = _width * _height;
	unsigned char* b = &_bits[0];
	for(int i=0; i<size; i++)
	{
		unsigned char tmp	= b[i*_nChannels];
		b[i*_nChannels]		= b[i*_nChannels + 2];
		b[i*_nChannels + 2] = tmp;
	}
}

void CFreeImage::createImage(int width, int height, int nChannels, const unsigned char* bits)
{
	_width = width;
	_height = height;
	_nChannels = nChannels;
	if(bits)
		_bits.assign(bits, bits + width*height*nChannels);
	else
		_bits.resize(width*height*nChannels, 0);
}

bool CFreeImage::load(const char* path)
{
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;

	fif = FreeImage_GetFileType(path, 0);
	int flag=0;
	if(fif == FIF_UNKNOWN) {
		fif = FreeImage_GetFIFFromFilename(path);
	}
	// check that the plugin has reading capabilities ...
	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)) {
		// ok, let's load the file
		FIBITMAP *dib = FreeImage_Load(fif, path, flag);
		if(dib==0){
			printf("load image failed: %s\n", path);
			return false;
		}
		_nChannels = FreeImage_GetInfo(dib)->bmiHeader.biBitCount/8;
		_width = FreeImage_GetWidth(dib);
		_height = FreeImage_GetHeight(dib);

		_bits.resize(_width*_height*_nChannels);

		//bitmap is size of 4n bytes per row.
		int ebit = (_width*_nChannels %4)==0 ? 0:1;
		int xsize=(_width*_nChannels/4 + ebit) *4;
		int rowRes = xsize - _width * _nChannels;
	

		const unsigned char *src = FreeImage_GetBits(dib);
		unsigned char *dst = &_bits[0];
		for(int i=0; i<_height; i++)
		{
			for(int j=0; j<_width; j++)
				for(int k=0; k<_nChannels; k++)
					*dst++ = *src++;
			src += rowRes;
		}

		FreeImage_Unload(dib);
	}
	else{
		printf("Unsupported image format!\n");
		return false;
	}
	return true;
}

bool CFreeImage::save(const char* path)const
{
	if (_bits.size()==0)
	{
		printf("Empty Image!\n");
		return false;
	}

	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFIFFromFilename(path);

	if(fif == FIF_UNKNOWN){
		printf("imwrite, unsupported format:%s\n", path);
		return false;
	}

	int ebit = (_width*_nChannels %4)==0 ? 0:1;
	int xsize=(_width*_nChannels/4 + ebit) *4;
	int res = xsize - _width*_nChannels;

	FIBITMAP *dib = FreeImage_Allocate(_width,_height,_nChannels*8);

	unsigned char *src=FreeImage_GetBits(dib);
	const unsigned char * dst = &_bits[0];
	for(int i=0; i<_height; i++)
	{
		for(int j=0; j<_width; j++)
			for(int k=0; k<_nChannels; k++)
				(*src++)=(*dst++);
		//added for bitmap format
		for(int j=0; j<res; j++)
			(*src++)=0;
	}

	FreeImage_Save(fif,dib,path);
	FreeImage_Unload(dib);
	return true;
}

static void BilinearImageResize(unsigned char* dstBits, const unsigned char* srcBits, int dstW, int dstH, int srcW, int srcH, int nChannel)
{
	float scalex = ((float)srcW)/dstW;
	float scaley = ((float)srcH)/dstH;

	#pragma omp parallel for
	for(int y=0; y<dstH; y++)
	{
		float dposy = y*scaley - 0.5f * (1.f-scaley);
		if(dposy < 0)
			dposy = 0.f;
		if(dposy > srcH-1)
			dposy = float(srcH-1);
		int iposy = int(dposy);
		float ydiff = dposy - iposy;
		unsigned char* pDst = dstBits + y*dstW*nChannel;
		const unsigned char* pSrc1 = srcBits + iposy*srcW*nChannel;
		const unsigned char* pSrc2 = srcBits + min(iposy+1,srcH-1)*srcW*nChannel;
		for(int x=0; x<dstW; x++)
		{
			float dposx = x*scalex - 0.5f * (1.f-scalex);
			if(dposx<0.f)
				dposx=0.f;
			if(dposx > srcW-1)
				dposx = float(srcW-1);
			int iposx = int(dposx);
			int iposx1 = min(iposx+1, srcW-1);
			float xdiff = dposx - iposx;
			float a00 = (1-xdiff)*(1-ydiff);
			float a01 = (1-xdiff)*ydiff;
			float a10 = xdiff*(1-ydiff);
			float a11 = xdiff*ydiff;
			const int x4 = x*nChannel;
			const int iposx4 = iposx*nChannel;
			const int iposx14 = iposx1*nChannel;
			for(int k=0; k<nChannel; k++)
				pDst[x4+k] = (unsigned char)(a00 * pSrc1[iposx4+k] + a01*pSrc2[iposx4+k] + a10*pSrc1[iposx14+k] + a11*pSrc2[iposx14+k]);
		}//end for x
	}//end for y
}

//lanczos3
inline double lanczos3(double x)
{
	double f = 0.0;
	const static double eps =  2.220446049250313e-16;
	double pix = 3.1415926535*x;
	f = (sin(pix) * sin(pix/3.0) + eps) / ( (pix*pix) / 3.0 + eps);
	f = f * (abs(x) < 3);
	return f;
}
void Lanczos3KernelIdx(int srcLen, int dstLen, std::vector<std::vector<float>>& weights, std::vector<std::vector<int>>& indices)
{
	float scale = float(srcLen)/float(dstLen);
	int kernel_size = 6;
	if(scale > 1.0) kernel_size = int(kernel_size*scale);
	weights.resize(dstLen);
	indices.resize(dstLen);
	for(int i=0; i<dstLen; i++)
	{
		const float dpos = i*scale - 0.5f * (1.f-scale);
		int ipos = int(dpos);
		if(dpos < 0) ipos--;
		std::vector<float>& w = weights[i];
		std::vector<int>& id = indices[i];
		w.resize(kernel_size);
		id.resize(kernel_size);
		float sumW = 0.f;
		const int left = ipos-kernel_size/2+1, right = ipos+kernel_size/2;
		for(int j=left; j<=right; j++)
		{
			if(scale > 1.f)
				w[j-left] = float(lanczos3((dpos - j)/scale));
			else
				w[j-left] = float(lanczos3(dpos - j));
			id[j-left] = max(0, min(srcLen-1,j));
			sumW += w[j-left];
		}
		for(int j=0; j<kernel_size; j++)
			w[j] /= sumW;
	}
}

void ImageResizeLanczos3RGBA(const CFreeImage& src, CFreeImage& dst)
{
	if(dst.width()==0 || dst.height()==0 || src.width()==0 || src.height()==0)
		return;
	std::vector<ldp::Float4> imTmp;
	imTmp.resize(dst.width() * src.height(), 0.f);

	//prepare weights and indices
	std::vector<std::vector<float>> weightsX, weightsY;
	std::vector<std::vector<int>> idxX, idxY;
	Lanczos3KernelIdx(src.width(), dst.width(), weightsX, idxX);
	Lanczos3KernelIdx(src.height(), dst.height(), weightsY, idxY);

	//x-direction scale
	//#pragma omp parallel for
	for(int y=0; y<src.height(); y++)
	{
		ldp::UChar4* pSrc= (ldp::UChar4*)src.rowPtr(y);
		ldp::Float4* pTmp= imTmp.data() + y*dst.width();
		for(int x=0; x<dst.width(); x++)
		{
			const std::vector<float>& weights = weightsX[x];
			const std::vector<int>& idx = idxX[x];
			for(size_t j=0; j<weights.size(); j++)
			{
				const int ij = idx[j];
				pTmp[x][0] += pSrc[ij][0] * weights[j];
				pTmp[x][1] += pSrc[ij][1] * weights[j];
				pTmp[x][2] += pSrc[ij][2] * weights[j];
				pTmp[x][3] += pSrc[ij][3] * weights[j];
			}
		}//end for x
	}//end for y

	//y-direction scale
	//#pragma omp parallel for
	for(int x=0; x<dst.width(); x++)
	{
		for(int y=0; y<dst.height(); y++)
		{
			const std::vector<float>& weights = weightsY[y];
			const std::vector<int>& idx = idxY[y];

			ldp::Float4 t = 0.f;
			for(size_t j=0; j<weights.size(); j++)
			{
				const ldp::Float4& pixelTmp = imTmp.data()[ idx[j]*dst.width() + x ];
				t[0] += pixelTmp[0] * weights[j];
				t[1] += pixelTmp[1] * weights[j];
				t[2] += pixelTmp[2] * weights[j];
				t[3] += pixelTmp[3] * weights[j];
			}
			ldp::UChar4* pixelDst = ((ldp::UChar4* )dst.rowPtr(y)) + x;
			pixelDst[0] = (unsigned char)min(255, max(0, int(t[0]+0.5f)));
			pixelDst[1] = (unsigned char)min(255, max(0, int(t[1]+0.5f)));
			pixelDst[2] = (unsigned char)min(255, max(0, int(t[2]+0.5f)));
			pixelDst[3] = (unsigned char)min(255, max(0, int(t[3]+0.5f)));
		}//end for y
	}//end for x
}

bool CFreeImage::ResizeImage(CFreeImage& dst, int w, int h, ResizeType type)const
{
	dst.createImage(w, h, nChannels());
	if(type == Bilinear)
		BilinearImageResize(dst.getBits(), getBits(), dst.width(), dst.height(), width(), height(), nChannels());
	else if(type == Lanczos3)
	{
		if(nChannels() != 4)
			return false;
		ImageResizeLanczos3RGBA(*this, dst);
	}
	return true;
}