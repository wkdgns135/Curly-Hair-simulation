#pragma once
#include "VectorCalcdevice.cuh"
#include "math_functions.h"
#include "vector_functions.h"

__global__ void array_init(float3 *arr) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	arr[tid] = make_float3(0.0, 0.0, 0.0);
}

__device__ void array_copy_k(float3 *a, float3 *b) {
	for (int i = 0; i < sizeof(a) / sizeof(float3); i++) {
		a[i].x = b[i].x;
		a[i].y = b[i].y;
		a[i].z = b[i].z;
	}
}

__device__  float3 vector_multiply_k(float3 a, float3 b) {
	float3 tmp;
	tmp.x = a.x * b.x;
	tmp.y = a.y * b.y;
	tmp.z = a.z * b.z;
	return tmp;
}


__device__  float3 vector_multiply_k(float3 a, double b) {
	float3 tmp;
	tmp.x = a.x * b;
	tmp.y = a.y * b;
	tmp.z = a.z * b;

	return tmp;
}

__device__  float3 vector_add_k(float3 a, float3 b) {
	float3 tmp;
	tmp.x = a.x + b.x;
	tmp.y = a.y + b.y;
	tmp.z = a.z + b.z;

	return tmp;
}


__device__ float3 vector_add_k(float3 a, double b) {
	float3 tmp;
	tmp.x = a.x + b;
	tmp.y = a.y + b;
	tmp.z = a.z + b;

	return tmp;
}

__device__ float3 vector_sub_k(float3 a, float3 b) {
	float3 tmp;
	tmp.x = a.x - b.x;
	tmp.y = a.y - b.y;
	tmp.z = a.z - b.z;

	return tmp;
}

__device__ float3 vector_sub_k(float3 a, double b) {
	float3 tmp;
	tmp.x = a.x - b;
	tmp.y = a.y - b;
	tmp.z = a.z - b;

	return tmp;
}

__device__ float3 operator + (float3 a, float3 &b) {
	float3 tmp;
	tmp.x = a.x + b.x;
	tmp.y = a.y + b.y;
	tmp.z = a.z + b.z;

	return tmp;
}

__device__ float3 operator + (float3 a, float &b) {
	float3 tmp;
	tmp.x = a.x + b;
	tmp.y = a.y + b;
	tmp.z = a.z + b;

	return tmp;
}

__device__ float3 operator - (float3 a, float3 &b) {
	float3 tmp;
	tmp.x = a.x - b.x;
	tmp.y = a.y - b.y;
	tmp.z = a.z - b.z;

	return tmp;
}

__device__ float3 operator - (float3 a, float &b) {
	float3 tmp;
	tmp.x = a.x - b;
	tmp.y = a.y - b;
	tmp.z = a.z - b;

	return tmp;
}
__device__ float3 operator * (float3 a, float3 &b) {
	float3 tmp;
	tmp.x = a.x * b.x;
	tmp.y = a.y * b.y;
	tmp.z = a.z * b.z;

	return tmp;
}

__device__ float3 operator * (float3 a, float &b) {
	float3 tmp;
	tmp.x = a.x * b;
	tmp.y = a.y * b;
	tmp.z = a.z * b;

	return tmp;
}

__device__ float3 operator / (float3 a, float3 &b) {
	float3 tmp;
	tmp.x = a.x / b.x;
	tmp.y = a.y / b.y;
	tmp.z = a.z / b.z;

	return tmp;
}

__device__ int3 operator + (int3 a, int3 b) {
	int3 tmp;
	tmp.x = a.x + b.x;
	tmp.y = a.y + b.y;
	tmp.z = a.z + b.z;
	return tmp;
}


__device__ double vector_length_k(float3 a) {
	return sqrt(float(a.x*a.x + a.y * a.y + a.z * a.z));
}

__device__ float3 vector_normalized_k(float3 a) {
	float3 tmp;
	double norm = vector_length_k(a);
	if (norm != 0) {
		tmp.x = a.x / norm;
		tmp.y = a.y / norm;
		tmp.z = a.z / norm;
	}
	return tmp;
}


__device__  float3	vector_cross_k(float3 a, float3 b) {
	float3 tmp;
	tmp.x = ((a.y*b.z) - (a.z*b.y));
	tmp.y = ((a.z*b.x) - (a.x*b.z));
	tmp.z = ((a.x*b.y) - (a.y*b.x));
	return tmp;
}

__device__ double vector_dot_k(float3 a, float3 b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

//
//__device__ float3 multiply_transpose_frame_k(Frame f, float3 e) {
//	float3 tmp;
//	tmp.x =
//		e.x * f.aim.x +
//		e.y * f.up.x +
//		e.z * f.cross.x;
//
//	tmp.y =
//		e.x * f.aim.y +
//		e.y * f.up.y +
//		e.z * f.cross.y;
//
//	tmp.z =
//		e.x * f.aim.z +
//		e.y * f.up.z +
//		e.z * f.cross.z;
//	return tmp;
//}
//
//__device__ float3 multiply_frame_k(Frame f, float3 e) {
//	float3 tmp;
//	tmp.x =
//		e.x * f.aim.x +
//		e.y * f.aim.y +
//		e.z * f.aim.z;
//
//	tmp.y =
//		e.x * f.up.x +
//		e.y * f.up.y +
//		e.z * f.up.z;
//
//	tmp.z =
//		e.x * f.cross.x +
//		e.y * f.cross.y +
//		e.z * f.cross.z;
//	return tmp;
//}