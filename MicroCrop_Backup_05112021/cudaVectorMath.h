#pragma once

#include <math.h>
#include "cuda_runtime.h"

// Useful constants
const double TOLERANCE = 1e-15;
const double INF = 1e+25;
const double PI = 3.14159265358979323846;


///////////////////////////////////////////////////////////////////
//    _____                          _     _                       
//   |  ___|  _   _   _ __     ___  | |_  (_)   ___    _ __    ___ 
//   | |_    | | | | | '_ \   / __| | __| | |  / _ \  | '_ \  / __|
//   |  _|   | |_| | | | | | | (__  | |_  | | | (_) | | | | | \__ \
//   |_|      \__,_| |_| |_|  \___|  \__| |_|  \___/  |_| |_| |___/
//                                                                 
//////////////////////////////////////////////////////////////////


// Checks if a value is zero
template <class T>
inline __host__ __device__ bool isZero(const T value, const T tolerance = 1e-15)
{
    if (value < tolerance && value >(-tolerance))
    {
        return true;
    }
    else
    {
        return false;
    }
}

// Checks if two values are equal
template <class T>
inline __host__ __device__ bool areEqual(const T valueA, const T valueB, const T tolerance = 1e-15)
{
    if (!isZero(valueB))
    {
        double difference = fabs(valueA / valueB - 1);
        return difference < tolerance;
    }
    else if (!isZero(valueA))
    {
        double difference = fabs(valueB / valueA - 1);
        return difference < tolerance;
    }
    else
    {
        return true;
    }
}


/////////////////////////////////////////////////////////////////////
//     ___                          _                       _       
//    / _ \  __   __   ___   _ __  | |   ___     __ _    __| |  ___ 
//   | | | | \ \ / /  / _ \ | '__| | |  / _ \   / _` |  / _` | / __|
//   | |_| |  \ V /  |  __/ | |    | | | (_) | | (_| | | (_| | \__ \
//    \___/    \_/    \___| |_|    |_|  \___/   \__,_|  \__,_| |___/
//                                                                  
////////////////////////////////////////////////////////////////////


// Operator overloading (+)
inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// Operator overloading (+=)
inline __host__ __device__ void operator+=(double3& a, double3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

// Operator overloading (- subtraction)
inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Operator overloading (-=)
inline __host__ __device__ void operator-=(double3& a, double3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

// Operator overloading (- negate)
inline __host__ __device__ double3 operator-(double3& a)
{
    return make_double3(-a.x, -a.y, -a.z);
}

// Operator overloading (vector * scalar)
inline __host__ __device__ double3 operator*(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}

// Operator overloading (scalar * vector)
inline __host__ __device__ double3 operator*(double b, double3 a)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}

// Operator overloading (*= scalar)
inline __host__ __device__ void operator*=(double3& a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

// Operator overloading (/ scalar divider)
inline __host__ __device__ double3 operator/(double3 a, double b)
{
    double3 result = make_double3(0.0, 0.0, 0.0);

    // Check if the divider is not zero
    if (!isZero(b))
    {
        result.x = a.x / b;
        result.y = a.y / b;
        result.z = a.z / b;
    }
    else
    {
        result.x = 1e+25;
        result.y = 1e+25;
        result.z = 1e+25;
    }

    return result;
}

// Operator overloading (/= scalar divider)
inline __host__ __device__ void operator/=(double3& a, double b)
{
    // Check if the divider is not zero
    if (!isZero(b))
    {
        a.x /= b;
        a.y /= b;
        a.z /= b;
    }
    else
    {
        a.x = 1e+25;
        a.y = 1e+25;
        a.z = 1e+25;
    }
}

// Operator overloading (==)
inline __host__ __device__ bool operator==(double3& a, double3& b)
{
    bool equalX = areEqual(a.x, b.x);
    bool equalY = areEqual(a.y, b.y);
    bool equalZ = areEqual(a.z, b.z);

    return (equalX && equalY && equalZ);
}

// Operator overloading (!=)
inline __host__ __device__ bool operator!=(double3& a, double3& b)
{
    return !(a == b);
}


//////////////////////////////////////////////////////////////////////////////////////
//   __     __                _                                          _     _     
//   \ \   / /   ___    ___  | |_    ___    _ __     _ __ ___     __ _  | |_  | |__  
//    \ \ / /   / _ \  / __| | __|  / _ \  | '__|   | '_ ` _ \   / _` | | __| | '_ \ 
//     \ V /   |  __/ | (__  | |_  | (_) | | |      | | | | | | | (_| | | |_  | | | |
//      \_/     \___|  \___|  \__|  \___/  |_|      |_| |_| |_|  \__,_|  \__| |_| |_|
//                                                                                   
/////////////////////////////////////////////////////////////////////////////////////


// Dot product
inline __host__ __device__ double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross product
inline __host__ __device__ double3 cross(double3 a, double3 b)
{
    return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Length
inline __host__ __device__ double length(double3 a)
{
    return sqrt(dot(a, a));
}

// Normalise
inline __host__ __device__ double3 get_normalize(double3 a)
{
    double3 result = make_double3(1e+25, 1e+25, 1e+25);

    double len = length(a);

    if (!isZero(len))
    {
        result.x = a.x / len;
        result.y = a.y / len;
        result.z = a.z / len;
    }

    return result;
}

// Angle between two vectors
inline __host__ __device__ double angle(double3 a, double3 b)
{
    double angle = 1e+25;

    double dot_product = dot(a, b);

    double length_a = length(a);
    double length_b = length(b);

    if (!isZero(length_a) && !isZero(length_b))
    {
        angle = acos(dot_product / (length_a * length_b));
    }

    return angle;
}


