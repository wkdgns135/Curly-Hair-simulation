#ifndef __VECTOR_CALC_H__
#define __VECTOR_CALC_H__

#pragma once
#include "HairModel.h"
#include "vector_types.h"

float3		operator * (float3 a, float3 b);
float3		operator * (float3 a, double b);
float3		operator + (float3 a, float3 b);
float3		operator + (float3 a, double b);
float3		operator - (float3 a, float3 b);
float3		operator - (float3 a, double b);
float3		operator / (float3 a, float3 b);
float3		operator / (float3 a, double b);
double		vector_length(float3 a);
void		vector_normalize(float3 &a);
float3		vector_normalized(float3 a);
float3		vector_cross(float3 a, float3 b);
double		vector_dot(float3 a, float3 b);
float3		multiply_transpose_frame(Frame f, float3 e);
float3		multiply_frame(Frame f, float3 e);

#endif