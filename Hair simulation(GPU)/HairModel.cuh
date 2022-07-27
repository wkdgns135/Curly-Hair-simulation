//#pragma once
//#include "vector_types.h"
//#include <cuda_runtime.h>
//#include <HairModel.h>
//
////__constant__ double	k_s;
////__constant__ double	c_s;
////__constant__ double	alpha_b;
////__constant__ double	k_b;
////__constant__ double	c_b;
////__constant__ double	alpha_c;
////__constant__ double	k_c;
////__constant__ double	c_c;
////__constant__ double	v_threshold;
////__constant__ double	s_threshold;
//
//struct __device_builtin__ Strand_d {
//	double3 *r_p_p_d;
//	double3 *r_s_p_p_d;
//	double r_p_l_d;
//	Frame *r_s_f_d;
//	double3 *t_d;
//};
//typedef __device_builtin__ struct Strand_d;
//
//__constant__ Strand_d *s_d;
//
//__global__ void integrate_internal_hair_force();
//__global__ void integrate_damping_force();
//__global__ void update_position();
//__global__ void smoothing_function();
//
//void cuda_init(Strand *s);
//void simulation();
//
