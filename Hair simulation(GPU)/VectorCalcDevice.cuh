#ifndef __VECTOR_CALC_K__
#define __VECTOR_CALC_K__

#include "vector_types.h"

__global__ void array_init(float3 *arr);

__device__ void array_copy_k(float3 *a, float3 *b);

__device__  float3 vector_multiply_k(float3 a, float3 b);

__device__  float3 vector_multiply_k(float3 a, double b);

__device__  float3 vector_add_k(float3 a, float3 b);

__device__ float3 vector_add_k(float3 a, double b);

__device__ float3 vector_sub_k(float3 a, float3 b);

__device__ float3 vector_sub_k(float3 a, double b);

__device__ float3 operator + (float3 a, float3 &b);
__device__ float3 operator + (float3 a, float &b);

__device__ float3 operator - (float3 a, float3 &b);
__device__ float3 operator - (float3 a, float &b);

__device__ float3 operator * (float3 a, float3 &b);
__device__ float3 operator * (float3 a, float &b);

__device__ float3 operator / (float3 a, float &b);



__device__ int3 operator + (int3 a, int3 b);

__device__ double vector_length_k(float3 a);

__device__ void vector_normalize_k(float3 &a);

__device__ float3 vector_normalized_k(float3 a);

__device__  float3	vector_cross_k(float3 a, float3 b);

__device__ double vector_dot_k(float3 a, float3 b);

//__device__ float3 multiply_transpose_frame_k(Frame f, float3 e);
//
//__device__ float3 multiply_frame_k(Frame f, float3 e);

#endif