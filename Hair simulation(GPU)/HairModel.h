#pragma once
#include <random>
#include <ctime>
#include <iostream>
#include <vector>
#include "vector_types.h"

#define K_S 500000.0
#define C_S 4000.0

#define K_B 30000.0
#define C_B 2000.0

#define K_C 15000.0
#define C_C 1000.0

#define A_B 1.0
#define A_C 1.0

using namespace std;

typedef struct Frame {
	float3 aim, up, cross;
}Frame;


class Sphere {
public:
	float3 pos;
	float radius;

public:
	void move_sphere(float3 dst);
};

class HairModel
{
public:
	unsigned int TOTAL_SIZE = 0;
	unsigned int MAX_SIZE = 0;
	unsigned int STRAND_SIZE = 0;
	vector<vector<float3>> v;
	
public://Coefficient

public: //Host
	int *p_i;

	double *r_p_l;
	float3 *p_p;
	float3 *p_v;
	float3 *s_p_p;
	float3 *r_p_p;
	float3 *r_s_p_p;
	float3 *s_p_v;
	Frame *r_s_f;
	Frame *s_f;
	float3 *t;
	float3 *d;

	//collision
	float3 sphere_pos;
	float sphere_radius;

public: //Device
	int *p_i_d;
	double *r_p_l_d;
	float3 *p_p_d;
	float3 *p_v_d;
	float3 *p_f_d;

	float3 *s_p_p_d;
	float3 *s_p_v_d;
	float3 *r_p_p_d;
	float3 *r_s_p_p_d;
	Frame *r_s_f_d;
	Frame *s_f_d;
	float3 *t_d;

public:
	void velocity_smoothing_function(float3 *lambda, float3 *dst,double *l, double alpha, bool is_position);
	void position_smoothing_function(float3 *lambda, float3 *dst, double *l, double alpha, bool is_position);
	void compute_frame(Frame *f, float3 *p);
	void array_copy(float3 *a, float3 *b);

public:
	void move_sphere(float3 dst);
	void move_root(int dst);
public:
	HairModel();
	void draw_point();
	void draw_wire();
	void draw_frame();

public:
	void device_info();
public:
	void device_init();
	void simulation();
};
