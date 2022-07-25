#pragma once
#include "vector_types.h"

//__constant__ double	k_s;
//__constant__ double	c_s;
//__constant__ double	alpha_b;
//__constant__ double	k_b;
//__constant__ double	c_b;
//__constant__ double	alpha_c;
//__constant__ double	k_c;
//__constant__ double	c_c;
//__constant__ double	v_threshold;
//__constant__ double	s_threshold;

__global__ void integrate_internal_hair_force();
__global__ void integrate_damping_force();
__global__ void update_position();
__global__ void smoothing_function();

