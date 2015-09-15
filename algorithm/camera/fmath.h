#ifndef __FMATH_H__
#define __FMATH_H__

#include <stddef.h>
#include <stdio.h>
#include <math.h>

class Vector2f;
class Vector3f;

namespace fmath {
	/** *************************************************************************************
	* Constants
	* ***************************************************************************************/

	/** pi-s */
#pragma push_macro("PI")
#undef PI
	const float PI			= 3.141592653589793f;
	const float TWO_PI		= 2 * PI;
	const float HALF_PI		= 0.5f * PI;
	const float QUARTER_PI	= 0.25f * PI;
	const float INV_PI		= 1.0f / PI;
	const float INV_TWO_PI	= 1.0f / TWO_PI;

	/** Converts of degree and radius */
	const float DEG_TO_RAD = PI / 180.0f;
	const float RAD_TO_DEG = 180.0f / PI;

	/** value of log(2) */
	const float LN2 = 0.693147180559945f;

	/** A "close to zero" float epsilon value for use*/
#undef FLT_EPSILON
#undef FLT_MIN
#undef FLT_MAX
	const float FLT_EPSILON = 1.1920928955078125E-7f;
	const float  FLT_MIN = 1.1754943e-038f;
	const float  FLT_MAX = 3.4028233e+038f;

	/** A close to zero threshold to use*/
	const float ZERO_TOLERANCE = 0.0001f;

	const float ONE_THIRD =1.f/3.f;

	/** *************************************************************************************
	* Declarations
	* **************************************************************************************/
	inline float sinef(const float& x);
	inline float cosinef(const float& x);

	inline bool cmpf(const float& a, const float& b);
	inline bool isZerof(const float& x);
	inline bool isOnef(const float& x);

	inline float absf(const float& x);
	inline float sqr(const float& x);

	//asm && Newton-iteration based algorithm
	inline float invSqrt(const float& x);

	inline bool isPowerOfTwo(const int& n);

	inline int nearestPowerOfTwo(const int& n);

	/** Linear interpolation from start value to end value
	* return: ((1-percent)*start)+percent*end
	*/
	inline float interpolate(const float& percent, const float &start, const float& end);

	/** Returns 1 if iValue>0, -1 if iValue<0, 0 otherwise*/
	inline int sign(int iValue);

	/**
     * @param x
     *            the value whose sign is to be adjusted.
     * @param y
     *            the value whose sign is to be used.
     * @return x with its sign changed to match the sign of y.
     */
	inline float copySign(const float& x, const float& y);

	 /**
     * Take a float input and clamp it between min and max.
     * 
     * @param input
     * @param min
     * @param max
     * @return clamped input
     */
    inline float clamp(const float& input, const float& min, const float& max);

	inline bool isInfinite(const float& val)
	{
		return val<FLT_MAX && val>FLT_MIN;
	}


	

	/** ***********************************************************************************
	* Implementations
	* *************************************************************************************/

	inline float absf(const float& x)
	{
		return x>0 ? x : -x;
	}

	inline float sqr(const float& x)
	{
		return x*x;
	}

	inline float sinef(const float& x)
	{
		return sin(x);
	}

	inline float cosinef(const float& x)
	{
		return cos(x);
	}

	inline bool isZerof(const float& x)
	{
		return fabs(x) < FLT_EPSILON;
	}

	inline bool isOnef(const float& x)
	{
		return fabs(x-1) < FLT_EPSILON;
	}

	inline bool cmpf(const float& a, const float& b)
	{
		return fabs(a-b) < FLT_EPSILON;
	}

	inline bool isPowerOfTwo(const int& n)
	{
		return (n>0) && (n & (n-1)) == 0;
	}

	inline int nearestPowerOfTwo(const int& n)
	{
		return (int) (pow(2, ceil(log((float)n))) / LN2);
	}

	inline float interpolate(const float& percent, const float &start, const float &end)
	{
		return (start == end) ? start : ( (1-percent)*start + percent*end );
	}

	inline int sign(int iValue)
	{
		return (iValue>0) ? 1 : ( (iValue<0) ? -1:0 );
	}

    inline float copysign(const float& x, const float& y) 
	{
        if (y >= 0 && x <= -0)
            return -x;
        else if (y < 0 && x >= 0)
            return -x;
        else
            return x;
    }

	inline float clamp(const float& input, const float& min, const float& max) 
	{
        return (input < min) ? min : (input > max) ? max : input;
    }

#pragma pop_macro("PI")
}; /*namespace fmath*/




#endif /*#define __FMATH_H__*/