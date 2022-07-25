#pragma once
#include <random>
#include <ctime>
#include <iostream>
#include "GL/glut.h"
#include "vector_types.h"
#define STRAND_SIZE 10
#define PARTICLE_SIZE 128


using namespace std;

struct Strand
{
	double r_p_l;
	double3 *p_p;
	double3 *s_p_p;
	double3 *r_p_p;
	double3 *r_s_p_p;
	double3 *t;
};

struct Frame {
	double3 x, y, z;
};

class HairModel
{
	Strand s[STRAND_SIZE];
public:
	HairModel();
	void draw_point();
	void draw_wire();

public:
	double3*  smoothing_function(double3 *lambda, double l, double alpha, bool is_position);
};


