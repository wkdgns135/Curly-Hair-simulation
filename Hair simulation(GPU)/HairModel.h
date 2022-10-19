#ifndef __HAIR_MODEL__
#define __HAIR_MODEL__

#pragma once
#include <random>
#include <ctime>
#include <iostream>
#include <vector>
#include "Particle.h"
#include "HashTableDevice.h"

#define K_S 500000.0
#define C_S 4000.0

#define K_B 30000.0
#define C_B 2000.0

#define K_C 15000.0
#define C_C 1000.0

#define A_B 0.23
#define A_C 1.0

using namespace std;

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

public:
	Particle particle_host;
	Particle particle_device;

	//collision
	float3 sphere_pos;
	float sphere_radius;
public: // added by jhkim
	HashTableDevice	_hashing;

public:
	void velocity_smoothing_function(float3 *lambda, float3 *dst, double *l, double alpha, bool is_position);
	void position_smoothing_function(float3 *lambda, float3 *dst, double *l, double alpha, bool is_position);
	void compute_frame(Frame *f, float3 *p);
	void array_copy(float3 *a, float3 *b);

public:
	void move_sphere(float3 dst);
	void move_root(int dst);
public:
	HairModel();
	//void draw_point();
	//void draw_wire();
	//void draw_frame();
public:
	// added by jhkim
	void	open(char *filename);
public:
	void device_info();
public:
	void device_init();
	void simulation();
public: // added by jhkim
	void	saveParticle(char *filename);
	void	updateHashing(void);
	void	freeHashing(void);
};

#endif // !__HAIR_MODEL__
