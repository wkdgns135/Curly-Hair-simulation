#pragma once
#include <random>
#include <ctime>
#include <iostream>
#include <vector>
#include "vector_types.h"

#define K_S 5000.0
#define C_S 200.0

#define K_B 500.0
#define C_B 3000.0

using namespace std;

typedef struct Frame {
	float3 aim, up, cross;
}Frame;


class HairModel
{
public:
	unsigned int TOTAL_SIZE = 0;
	unsigned int MAX_SIZE = 0;
	unsigned int STRAND_SIZE = 0;
	vector<vector<float3>> v;

public: //Host
	int *p_i;

	double *r_p_l;
	float3 *p_p;
	float3 *s_p_p;
	float3 *r_p_p;
	float3 *r_s_p_p;
	Frame *r_s_f;
	Frame *s_f;
	float3 *t;

public: //Device
	int *p_i_d;
	double *r_p_l_d;
	float3 *p_p_d;
	float3 *p_v_d;
	float3 *p_f_d;

	float3 *s_p_p_d;
	float3 *r_p_p_d;
	float3 *r_s_p_p_d;
	Frame *r_s_f_d;
	Frame *s_f_d;
	float3 *t_d;

	float3* smoothing_function(float3 *lambda, double *l, double alpha, bool is_position);

	void compute_frame(Frame *f, float3 *p);
	void array_copy(float3 *a, float3 *b);

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
