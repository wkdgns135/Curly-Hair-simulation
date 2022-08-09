#ifndef __VECTOR_CALC_K__
#define __VECTOR_CALC_K__

#include "vector_types.h"
#include "math_functions.h"

__global__ void array_init(double3 *arr) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	arr[tid] = make_double3(0.0, 0.0, 0.0);
}

__device__ void array_copy_k(double3 *a, double3 *b) {
	for (int i = 0; i < PARTICLE_SIZE; i++) {
		a[i].x = b[i].x;
		a[i].y = b[i].y;
		a[i].z = b[i].z;
	}
}

__device__  double3 vector_multiply_k(double3 a, double3 b) {
	double3 tmp;
	tmp.x = a.x * b.x;
	tmp.y = a.y * b.y;
	tmp.z = a.z * b.z;
	return tmp;
}


__device__  double3 vector_multiply_k(double3 a, double b) {
	double3 tmp;
	tmp.x = a.x * b;
	tmp.y = a.y * b;
	tmp.z = a.z * b;

	return tmp;
}

__device__  double3 vector_add_k(double3 a, double3 b) {
	double3 tmp;
	tmp.x = a.x + b.x;
	tmp.y = a.y + b.y;
	tmp.z = a.z + b.z;

	return tmp;
}


__device__ double3 vector_add_k(double3 a, double b) {
	double3 tmp;
	tmp.x = a.x + b;
	tmp.y = a.y + b;
	tmp.z = a.z + b;

	return tmp;
}

__device__ double3 vector_sub_k(double3 a, double3 b) {
	double3 tmp;
	tmp.x = a.x - b.x;
	tmp.y = a.y - b.y;
	tmp.z = a.z - b.z;

	return tmp;
}

__device__ double3 vector_sub_k(double3 a, double b) {
	double3 tmp;
	tmp.x = a.x - b;
	tmp.y = a.y - b;
	tmp.z = a.z - b;

	return tmp;
}

__device__ double vector_length_k(double3 a) {
	return sqrt(float(a.x*a.x + a.y * a.y + a.z * a.z));
}

__device__ void vector_normalize_k(double3 &a) {
	double norm = vector_length_k(a);
	if (norm != 0) {
		a.x = a.x / norm;
		a.y = a.y / norm;
		a.z = a.z / norm;
	}
}

__device__ double3 vector_normalized_k(double3 a) {
	double3 tmp;
	double norm = vector_length_k(a);
	if (norm != 0) {
		tmp.x = a.x / norm;
		tmp.y = a.y / norm;
		tmp.z = a.z / norm;
	}
	return tmp;
}


__device__  double3	vector_cross_k(double3 a, double3 b) {
	double3 tmp;
	tmp.x = ((a.y*b.z) - (a.z*b.y));
	tmp.y = ((a.z*b.x) - (a.x*b.z));
	tmp.z = ((a.x*b.y) - (a.y*b.x));
	return tmp;
}

__device__ double vector_dot_k(double3 a, double3 b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__global__ void compute_frame_k(Frame *f, double3 *p) {
	double3 aim = vector_sub_k(p[threadIdx.x * PARTICLE_SIZE +  1], p[threadIdx.x * PARTICLE_SIZE + 0]);
	vector_normalize_k(aim);

	double3 up;
	up.x = aim.z - aim.y;
	up.y = aim.x - aim.z;
	up.z = aim.y - aim.x;

	vector_normalize_k(up);
	for (int i = 1; i < PARTICLE_SIZE - 1; i++) {
		int index = threadIdx.x * PARTICLE_SIZE + i;
		double3 aim = vector_sub_k(p[index + 1], p[index]);
		vector_normalize_k(aim);

		double3 cross = vector_cross_k(aim, up);
		vector_normalize_k(cross);

		up = vector_cross_k(cross, aim);
		vector_normalize_k(up);

		f[index].aim.x = aim.x;
		f[index].aim.y = up.x;
		f[index].aim.z = cross.x;

		f[index].up.x = aim.y;
		f[index].up.y = up.y;
		f[index].up.z = cross.y;

		f[index].cross.x = aim.z;
		f[index].cross.y = up.z;
		f[index].cross.z = cross.z;

	}
}

__device__ double3 multiply_transpose_frame_k(Frame f, double3 e) {
	double3 tmp;
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

__device__ double3 multiply_frame_k(Frame f, double3 e) {
	double3 tmp;
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

//__device__ double3* smoothing_function_k(double3 *lambda, int *p_i, double *l, double alpha, bool is_position) {
//	double beta = 0.0;
//
//	double3  d[STRAND_SIZE * PARTICLE_SIZE];
//	double3 pos[STRAND_SIZE * PARTICLE_SIZE];
//	//lambda가 파티클 위치일 경우 return하기위한 pos vector
//
//	array_copy_k(d, lambda);
//
//	//beta formulation, l = 파티클간의 평균길이
//
//	int index = threadIdx.x * PARTICLE_SIZE;
//	d[index] = vector_sub_k(lambda[index + 1], lambda[index]);
//	beta = 1 > 1 - exp(-l[threadIdx.x] / alpha) ? 1 - exp(-l[threadIdx.x] / alpha) : 1;
//
//	for (int j = 1; j < PARTICLE_SIZE - 1; j++) {
//		int index_1 = j - 1 >= 0 ? j - 1 : 0;
//		int index_2 = j - 2 >= 0 ? j - 2 : 0;
//
//		int index1 = threadIdx.x * PARTICLE_SIZE + index_1;
//		int index2 = threadIdx.x * PARTICLE_SIZE + index_2;
//		index = threadIdx.x * PARTICLE_SIZE + j;
//
//		double3 term1 = vector_multiply_k(d[index_1], 2 * (1 - beta));
//		double3 term2 = vector_multiply_k(d[index_2], ((1 - beta) * (1 - beta)));
//		double3 term3 = vector_sub_k(term1, term2);
//		double3 term4 = vector_multiply_k(vector_sub_k(lambda[index + 1], lambda[index]), (beta * beta));
//		d[index] = vector_add_k(term3, term4);
//	}
//	
//
//	if (is_position) {
//		int index = threadIdx.x * PARTICLE_SIZE;
//		pos[index] = lambda[index];
//		for (int j = 1; j < PARTICLE_SIZE; j++) {
//			index = threadIdx.x * PARTICLE_SIZE + j;
//			pos[index] = vector_add_k(d[index - 1], pos[index - 1]);
//		}
//
//		return pos;
//	}
//	return d;
//}

__global__ void smoothing_function_k(double3 *lambda, double3 *dst, double *l, double alpha, bool is_position) {
	double beta = 0.0;

	array_copy_k(dst, lambda);

	//beta formulation, l = 파티클간의 평균길이
	int index = threadIdx.x * PARTICLE_SIZE;
	dst[index] = vector_sub_k(lambda[index + 1], lambda[index]);
	beta = 1 > 1 - exp(-l[threadIdx.x] / alpha) ? 1 - exp(-l[threadIdx.x] / alpha) : 1;
	for (int j = 1; j < PARTICLE_SIZE - 1; j++) {
		int index_1 = j - 1 >= 0 ? j - 1 : 0;
		int index_2 = j - 2 >= 0 ? j - 2 : 0;

		int index1 = threadIdx.x * PARTICLE_SIZE + index_1;
		int index2 = threadIdx.x * PARTICLE_SIZE + index_2;
		index = threadIdx.x * PARTICLE_SIZE + j;

		double3 term1 = vector_multiply_k(dst[index_1], 2 * (1 - beta));
		double3 term2 = vector_multiply_k(dst[index_2], ((1 - beta) * (1 - beta)));
		double3 term3 = vector_sub_k(term1, term2);
		double3 term4 = vector_multiply_k(vector_sub_k(lambda[index + 1], lambda[index]), (beta * beta));
		dst[index] = vector_add_k(term3, term4);
	}

	if (is_position) {
		double3 pos[PARTICLE_SIZE];
		array_copy_k(pos, dst);
		int index = threadIdx.x * PARTICLE_SIZE;
		dst[index] = lambda[index];
		for (int j = 1; j < PARTICLE_SIZE; j++) {
			index = threadIdx.x * PARTICLE_SIZE + j;
			dst[index] = vector_add_k(pos[index - 1], dst[index - 1]);
		}
	}
}
#endif