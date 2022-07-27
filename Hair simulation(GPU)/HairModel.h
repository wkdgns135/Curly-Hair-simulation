#pragma once
#include <random>
#include <ctime>
#include <iostream>
#include "GL/glut.h"
#include "vector_types.h"
#define STRAND_SIZE 1
#define PARTICLE_SIZE 128


using namespace std;
struct  Frame {
	double3 aim, up, cross;
};

struct Strand
{
	double r_p_l;
	double3 *p_p;
	double3 *s_p_p;
	double3 *r_p_p;
	double3 *r_s_p_p;
	Frame *r_s_f;
	Frame *s_f;
	double3 *t;
};


class HairModel
{
	Strand s[STRAND_SIZE];
public:
	HairModel();
	void draw_point();
	void draw_wire();
	void draw_frame();

public:
	double3*  smoothing_function(double3 *lambda, double l, double alpha, bool is_position);
};


