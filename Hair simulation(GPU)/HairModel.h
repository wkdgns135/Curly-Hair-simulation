#pragma once
#include <random>
#include <ctime>
#include <iostream>
#include "GL/glut.h"
#include "vector_types.h"

#define STRAND_SIZE 10
#define PARTICLE_SIZE 128


typedef struct Frame {
	double3 aim, up, cross;
}Frame;

using namespace std;

class HairModel
{
public:

	
public: //Host
	double *r_p_l;
	double3 *p_p;
	double3 *s_p_p;
	double3 *r_p_p;
	double3 *r_s_p_p;
	Frame *r_s_f;
	Frame *s_f;
	double3 *t;

public: //Device
	double *r_p_l_d;
	double3 *p_p_d;
	double3 *p_v_d;
	double3 *p_f_d;

	double3 *s_p_p_d;
	double3 *r_p_p_d;
	double3 *r_s_p_p_d;
	Frame *r_s_f_d;
	Frame *s_f_d;
	double3 *t_d;

public:
	HairModel();
	void draw_point();
	void draw_wire();
	void draw_frame();

public:
	double3*  smoothing_function(double3 *lambda, double *l, double alpha, bool is_position);

public:
	void device_init();

	void simulation();
};
