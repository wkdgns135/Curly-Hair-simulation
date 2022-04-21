#pragma once
#include "Particle.h"
#include "vector"

using namespace std;

class HairModel
{
public:
	Particle		*particle;
	Particle		*smoothed_particle;
	Particle		*rest_particle;
	Particle		*smoothed_rest_particle;
	vector<int>		size;
public:
public:
	double			k_s			= 500000;
	double			c_s			= 4470;  //;
	double			alpha_b		= 0.1;//0.1;
	double			k_b			= 30000;//30000;
	double			c_b			= 7500;// 5000;
	double			alpha_c		= 1;
	double			k_c			= 30000;// 15000;
	double			c_c			= 2000;// 100;
	double			v_threshold = 1;
	double			s_threshold = 1;
public:
					HairModel();
	void			init(Particle *p);
	void			pre_compute();
	void			simulation();

public:
	void			draw_wire(vector<vector<Vector3d>>);
	void			draw_point(vector<vector<Vector3d>>);
	void			draw_frame(Particle *p);
	void			move_root_particle(Vector3d dest);

public:
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

	vector<vector<Vector3d>> 		smoothing_function(vector<vector<Vector3d>> lambda, vector<double> l, double alpha, bool is_position);
};

