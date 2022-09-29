#pragma once
#include "Particle.h"
#include "vector"
#include <random>
#include <ctime>

using namespace std;

class HairModel
{
public:
	Particle		*particle;
	Particle		*smoothed_particle;
	Particle		*rest_particle;
	Particle		*smoothed_rest_particle;
	vector<int>		size;
	Vector3f		force;
public:
	float			k_s			= 500000;
	float			c_s			= 4000;
	float			alpha_b		= 5;
	float			k_b			= 30000;
	float			c_b			= 2000;
	float			alpha_c		= 1;
	float			k_c			= 15000;
	float			c_c			= 1000;
	float			v_threshold = 1;
	float			s_threshold = 1;
	float			total_wet	= 1;
	float			w_c		= 150;
	float			w_d		= 0.8;

public:
	//Vector3f	sphere = Vector3f(0, -12, -25);
	Vector3f	sphere = Vector3f(0, -30, -25);
	float		radius = 4;
public:
					HairModel();
	void			init(Particle *p);
	void			pre_compute();
	void			helix_function(Particle *p);
	void			simulation(Vector3f _force = Vector3f(0,0,0));
public:
	void			draw_wire(vector<vector<Vector3f>>);
	void			draw_point(vector<vector<Vector3f>>);
	void			draw_frame(Particle *p);
	void			move_root_particle(Vector3f dest);
	void			bouncing_test(float n);
public:
	//void			wetting_function(float n);
	void			collision_detect();
	void			integrate_internal_hair_force(); //include springs
	void			integrate_external_force();
	void			integrate_damping_force();
	void			update_position();

public:
	void			stretch_spring_force(int i, int j);
	void			bending_spring_force(int i, int j);
	void			core_spring_force(int i, int j);

	void			stretch_damping_force(int i, int j);
	void			bending_damping_force(int i, int j);
	void			core_damping_force(int i, int j);

	void			wet_force(int i, int j);

	vector<vector<Vector3f>> 		smoothing_function(vector<vector<Vector3f>> lambda, vector<float> l, float alpha, bool is_position);
};

