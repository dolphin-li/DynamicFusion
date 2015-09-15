#ifndef __ERROR_H__
#define __ERROR_H__
#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <windows.h>
#include <cstdarg>
#pragma   warning(disable:4290)

#ifdef _DEBUG
#define LDP_DEBUG_CHECK(express,errorType,msg) \
	if(!(express)) throw ldp::errorType(msg).show();
#else
#define LDP_DEBUG_CHECK(express,errorType,msg) \
	
#endif


namespace ldp{

	inline void safeDelete(void* p)
	{
		delete []p;
		p = 0;
	}
class Logger{
public:
	static int info(const char* string, ...)
	{
		::SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),FOREGROUND_GREEN|FOREGROUND_INTENSITY);
		va_list ap;
		int r;
		va_start (ap, string);
		r = vprintf (string, ap);
		va_end (ap);
		::SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),FOREGROUND_RED|FOREGROUND_BLUE|FOREGROUND_GREEN);
		return r;
	}
	static int warning(const char* string, ...)
	{
		::SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),FOREGROUND_GREEN|FOREGROUND_RED);
		va_list ap;
		int r;
		va_start (ap, string);
		r = vprintf (string, ap);
		va_end (ap);
		::SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),FOREGROUND_RED|FOREGROUND_BLUE|FOREGROUND_GREEN);
		return r;
	}
	static int error(const char* string, ...)
	{
		::SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),FOREGROUND_RED|FOREGROUND_INTENSITY);
		va_list ap;
		int r;
		va_start (ap, string);
		r = vprintf (string, ap);
		va_end (ap);
		::SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),FOREGROUND_RED|FOREGROUND_BLUE|FOREGROUND_GREEN);	
		return r;
	}
};

//===========================
class Error{
public:
	std::string msg;
public:
	Error(){
		msg.append("Error:\n");
	}
	Error(std::string s){
		msg.append("Error:\n");
		msg.append(s);
	}
	virtual Error& show()
	{
		Logger::error(msg.c_str());
		return *this;
	}
	virtual Error& show(int row, int col)
	{
		Logger::error("%s(%d,%d)\n",msg.c_str(),row,col);
		return *this;
	}
	virtual Error& show(int pos)
	{
		Logger::error("%s(%d)\n",msg.c_str(),pos);
		return *this;
	}
	virtual Error& show(const char* string,...)
	{
		::SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),FOREGROUND_RED|FOREGROUND_INTENSITY);
		va_list ap;
		int r;
		va_start (ap, string);
		r = vprintf (string, ap);
		va_end (ap);
		::SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),FOREGROUND_RED|FOREGROUND_BLUE|FOREGROUND_GREEN);	
		return *this;
	}
};//class Error

//===========================
class DivideByZeroError:public Error{
public:
	DivideByZeroError(){
		msg.clear();
		msg.append("DivideByZeroError:\n");
	}
	DivideByZeroError(std::string s){
		msg.clear();
		msg.append("DivideByZeroError:\n");
		msg.append(s);
	}
};//class DivideByZeroError

//===========================
class ArrayOutOfIndexError:public Error{
public:
	ArrayOutOfIndexError(){
		msg.clear();
		msg.append("ArrayOutOfIndexError:\n");
	}
	ArrayOutOfIndexError(std::string s){
		msg.clear();
		msg.append("ArrayOutOfIndexError:\n");
		msg.append(s);
	}
};//class ArrayOutOfIndexError

//===========================
class MatrixSizeMatchError:public Error{
public:
	MatrixSizeMatchError(){
		msg.clear();
		msg.append("MatrixSizeMatchError:\n");
	}
	MatrixSizeMatchError(std::string s){
		msg.clear();
		msg.append("MatrixSizeMatchError:\n");
		msg.append(s);
	}
};//class MatrixSizeMatchError

//===========================
class SinglarMatrixError:public Error{
public:
	SinglarMatrixError(){
		msg.clear();
		msg.append("SinglarMatrixError:\n");
	}
	SinglarMatrixError(std::string s){
		msg.clear();
		msg.append("SinglarMatrixError:\n");
		msg.append(s);
	}
};//class MatrixSizeMatchError

//===========================
class NullPointerError:public Error{
public:
	NullPointerError(){
		msg.clear();
		msg.append("NullPointerError:\n");
	}
	NullPointerError(std::string s){
		msg.clear();
		msg.append("NullPointerError:\n");
		msg.append(s);
	}
};//class NullPointerError

//===========================
class IOError:public Error{
public:
	IOError(){
		msg.clear();
		msg.append("IOError:\n");
	}
	IOError(std::string s){
		msg.clear();
		msg.append("IOError:\n");
		msg.append(s);
	}
};//class IOError

//===========================
class IntOverflowError:public Error{
public:
	IntOverflowError(){
		msg.clear();
		msg.append("IntOverflowError:\n");
	}
	IntOverflowError(std::string s){
		msg.clear();
		msg.append("IntOverflowError:\n");
		msg.append(s);
	}
};//class IntOverflowError

//===========================
class UnimplementedError:public Error{
public:
	UnimplementedError(){
		msg.clear();
		msg.append("UnimplementedError:\n");
	}
	UnimplementedError(std::string s){
		msg.clear();
		msg.append("UnimplementedError:\n");
		msg.append(s);
	}
};//class UnimplementedError

//===========================
class BadMeshError:public Error{
public:
	BadMeshError(){
		msg.clear();
		msg.append("BadMeshError:\n");
	}
	BadMeshError(std::string s){
		msg.clear();
		msg.append("BadMeshError:\n");
		msg.append(s);
	}
};//class BadMeshError

//===========================
class VmlError:public Error{
public:
	VmlError(){
		msg.clear();
		msg.append("VmlError:\n");
	}
	VmlError(std::string s){
		msg.clear();
		msg.append("VmlError:\n");
		msg.append(s);
		msg.append(", ");
	}
	VmlError& show(int errType)
	{
		switch(errType)
		{
		case -1:
			msg.append("VML_STATUS_BADSIZE\n");
			break;
		case -2:
			msg.append("VML_STATUS_BADMEM:Null pointer is passed\n");
			break;
		case 1:
			msg.append("VML_STATUS_ERRDOM:Array Elements Out of defination of the corresponding function\n");
			break;
		case 2:
			msg.append("VML_STATUS_SING: Divide-by-zero in at least one of elements\n");
			break;
		case 3:
			msg.append("VML_STATUS_OVERFLOW\n");
			break;
		case 4:
			msg.append("VML_STATUS_UNDERFLOW\n");
			break;
		default:
			msg.append("Not a typical error\n");
			break;
		}
		Logger::error(msg.c_str());
		return *this;
	}
};//class VmlError

};//namespace ldp


#endif//__ERROR_H__