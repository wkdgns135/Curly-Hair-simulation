#pragma once
#include "vector"
#include <random>
#include <ctime>
#include <iostream>
#include "HairModel.cuh"
#include "vector_calc.h"
#define STRAND_SIZE 100
#define PARTICLE_SIZE 128


using namespace std;


class HairModel
{
	Strand s[STRAND_SIZE];
public:
	HairModel();
	void pre_compute();
};

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