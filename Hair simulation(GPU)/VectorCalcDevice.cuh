#ifndef __VECTOR_CALC_K__
#define __VECTOR_CALC_K__

#include "vector_types.h"

struct matrix3
{
	float3 x, y, z;
};
__global__ void array_init(float3 *arr);
__device__ __host__ void array_copy_k(float3 *a, float3 *b);
__device__ __host__  float3 vector_multiply_k(float3 a, float3 b);
__device__ __host__  float3 vector_multiply_k(float3 a, double b);
__device__ __host__  float3 vector_add_k(float3 a, float3 b);
__device__ __host__ float3 vector_add_k(float3 a, double b);
__device__ __host__ float3 vector_sub_k(float3 a, float3 b);
__device__ __host__ float3 vector_sub_k(float3 a, double b);
__device__ __host__ float3 operator + (float3 a, float3 &b);
__device__ __host__ float3 operator + (float3 a, float &b);
__device__ __host__ float3 operator - (float3 a, float3 &b);
__device__ __host__ float3 operator - (float3 a, float &b);
__device__ __host__ float3 operator * (float3 a, float3 &b);
__device__ __host__ float3 operator * (float3 a, float &b);
__device__ __host__ float3 operator / (float3 a, float3 &b);
__device__ __host__ int3 operator + (int3 a, int3 b);
__device__ __host__ double vector_length_k(float3 a);
__device__ __host__ float3 vector_normalized_k(float3 a);
__device__ __host__  float3	vector_cross_k(float3 a, float3 b);
__device__ __host__ double vector_dot_k(float3 a, float3 b);
__device__ __host__ matrix3 operator +(matrix3 a, matrix3 b);
__device__ __host__ matrix3 operator *(matrix3 a, matrix3 b);
__device__ __host__ matrix3 operator *(matrix3 a, float b);
__device__ __host__ float3 rot_vec_by_mat(float3 a, matrix3 b);
__device__ __host__ matrix3 rot_mat_from_two_vectors(float3 a, float3 b);
__device__ __host__ void set_identity(matrix3 &a);
__device__ __host__ float mat_index(matrix3 mat, int i, int j);
//__device__ float3 multiply_transpose_frame_k(Frame f, float3 e);
//
//__device__ float3 multiply_frame_k(Frame f, float3 e);

#endif