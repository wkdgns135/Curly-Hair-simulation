#ifndef __VECTOR_CALC_K__
#define __VECTOR_CALC_K__

#include "vector_types.h"
#include "math_functions.h"

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

__device__ double vector_length_k(float3 a) {
	return sqrt(float(a.x*a.x + a.y * a.y + a.z * a.z));
}

__device__ void vector_normalize_k(float3 &a) {
	double norm = vector_length_k(a);
	if (norm != 0) {
		a.x = a.x / norm;
		a.y = a.y / norm;
		a.z = a.z / norm;
	}
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

__device__ float3 multiply_transpose_frame_k(Frame f, float3 e) {
	float3 tmp;
	tmp.x =
		e.x * f.aim.x +
		e.y * f.up.x +
		e.z * f.cross.x;

	tmp.y =
		e.x * f.aim.y +
		e.y * f.up.y +
		e.z * f.cross.y;

	tmp.z =
		e.x * f.aim.z +
		e.y * f.up.z +
		e.z * f.cross.z;
	return tmp;
}

__device__ float3 multiply_frame_k(Frame f, float3 e) {
	float3 tmp;
	tmp.x =
		e.x * f.aim.x +
		e.y * f.aim.y +
		e.z * f.aim.z;

	tmp.y =
		e.x * f.up.x +
		e.y * f.up.y +
		e.z * f.up.z;

	tmp.z =
		e.x * f.cross.x +
		e.y * f.cross.y +
		e.z * f.cross.z;
	return tmp;
}

__global__ void smoothing_function_k(float3 *lambda, float3 *dst, double *l, double alpha, bool is_position, int x, int y) {
	double beta = 0.0;

	float3  *d;
	float3 *pos;
	//lambda가 파티클 위치일 경우 return하기위한 pos vector

	array_copy_k(d, lambda);
	//beta formulation, l = 파티클간의 평균길이

	int index = threadIdx.x * y;

	d[index] = vector_sub_k(lambda[index + 1], lambda[index]);
	beta = 1 > 1 - exp(-l[threadIdx.x] / alpha) ? 1 - exp(-l[threadIdx.x] / alpha) : 1;

	for (int j = 1; j < y - 1; j++) {
		int index_1 = j - 1 >= 0 ? j - 1 : 0;
		int index_2 = j - 2 >= 0 ? j - 2 : 0;

		int index1 = threadIdx.x * y + index_1;
		int index2 = threadIdx.x * y + index_2;
		index = threadIdx.x * y + j;

		float3 term1 = vector_multiply_k(d[index_1], 2 * (1 - beta));
		float3 term2 = vector_multiply_k(d[index_2], ((1 - beta) * (1 - beta)));
		float3 term3 = vector_sub_k(term1, term2);
		float3 term4 = vector_multiply_k(vector_sub_k(lambda[index + 1], lambda[index]), (beta * beta));
		d[index] = vector_add_k(term3, term4);
	}
	

	if (is_position) {
		int index = threadIdx.x * y;
		pos[index] = lambda[index];
		for (int j = 1; j < y; j++) {
			index = threadIdx.x * y + j;
			pos[index] = vector_add_k(d[index - 1], pos[index - 1]);
		}

		return;
	}
	return;
}

#endif