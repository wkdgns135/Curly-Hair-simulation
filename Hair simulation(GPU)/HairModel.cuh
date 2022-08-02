#pragma once
#include <cuda_runtime.h>
#include <HairModel.h>

class HairModel_d {
public:
	double *r_p_l;
	double3 *p_p;
	double3 *p_v;
	double3 *p_f;

	double3 *s_p_p;
	double3 *r_p_p;
	double3 *r_s_p_p;
	Frame *r_s_f;
	Frame *s_f;
	double3 *t;
public:
	HairModel_d(HairModel h);
	double3* simulation();
};

__global__ void test_k(double3 *p_p);
__global__ void simulation_k(double3 *p_p, double3 *s_p_p, double3 *r_p_p, double3 *r_s_p_p, Frame *r_s_f, Frame *s_f, double3 *t);
__device__ void integrate_internal_hair_force();
__device__ void integrate_damping_force();
__device__ void update_position();
__device__ void smoothing_function();


