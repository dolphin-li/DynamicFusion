#ifndef __IMAGE_LOADER_H__
#define __IMAGE_LOADER_H__

#include <vector>
class CFreeImage
{
public:
	CFreeImage();
	CFreeImage(const CFreeImage& other);
	~CFreeImage();

	bool load(const char* filename);
	bool save(const char* filename)const;

	int width()const{return _width;}
	int height()const{return _height;}
	int nChannels()const{return _nChannels;}

	/**
	* if numChannels == 3 or 4, then return BGR or BGRA format
	* */
	const unsigned char* getBits()const{return _bits.data();}
	unsigned char* getBits(){return _bits.data();}
	const unsigned char* rowPtr(int y)const{return _bits.data()+y*_width*_nChannels;}

	/**
	* if numChannels == 3 or 4, then return BGR or BGRA format
	* */
	void createImage(int width, int height, int nChannels, const unsigned char* bits=0);

	enum ResizeType
	{
		Bilinear,
		Lanczos3,//bugs now...
	};
	bool ResizeImage(CFreeImage& dst, int w, int h, ResizeType type)const;

	void rgb2bgr();

	void clear();
private:
	std::vector<unsigned char> _bits; 
	int _width, _height, _nChannels;
};//ImageLoader










#endif//__IMAGE_LOADER_H__