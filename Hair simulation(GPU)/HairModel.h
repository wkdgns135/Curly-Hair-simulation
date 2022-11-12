#ifndef __HAIR_MODEL__
#define __HAIR_MODEL__

#pragma once
#include <random>
#include <ctime>
#include <iostream>
#include <vector>
#include "Particle.h"
#include "HashTableDevice.h"

using namespace std;

struct Params
{
	float	K_S;
	float	C_S;
	
	float	K_B;
	float	C_B;

	float	K_C;
	float	C_C;

	float	A_B;
	float	A_C;

	float	R_C;

	int3 grid_size;
	float3 cell_size;
	float particle_radius;
};

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
	float3 min_b, max_b;
	
public:
	vector<vector<uint>> indices;
	float3 *colors;
	float3 color;
	void	get_colors();
public:
	Particle particle_host;
	Particle particle_device;
	Params params_host;

	//collision
	float3 sphere_pos;
	float sphere_radius;
public: // added by jhkim
	HashTableDevice	_hashing;

public:
	void pre_compute();
	void velocity_smoothing_function(float3 *lambda, float3 *dst, double *l, double alpha, bool is_position);
	void position_smoothing_function(float3 *lambda, float3 *dst, double *l, double alpha, bool is_position);
	void compute_frame(Frame *f, float3 *p);
	void array_copy(float3 *a, float3 *b);
	void normalize_position();

public:
	void move_sphere(float3 dst);
	void move_root(int dst);
public:
	HairModel(char *filename);
	~HairModel();
	void draw_vertex(float3 v);
	void draw_point();
	void draw_wire();
	void test_draw();
	void draw_frame();
public:
	// added by jhkim
	void	open(char *filename);
public:
	void device_info();
public:
	void params_init();
	void device_init();
	void device_free();
	void simulation();
public: // added by jhkim
	void	saveParticle(char *filename);
	void	updateHashing(void);
	void	freeHashing(void);
};

#endif // !__HAIR_MODEL__
