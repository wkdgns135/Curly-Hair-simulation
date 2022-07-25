#include "HairModel.h"

HairModel::HairModel() {
	for (int i = 0; i < STRAND_SIZE; i++) {
		s[i].p_p = (double3*)malloc(12 * PARTICLE_SIZE); //double3 buffer size = 12
		s[i].s_p_p = (double3*)malloc(12 * PARTICLE_SIZE);
		s[i].r_p_p = (double3*)malloc(12 * PARTICLE_SIZE);
		s[i].r_s_p_p = (double3*)malloc(12 * PARTICLE_SIZE);
		s[i].t = (double3*)malloc(12 * PARTICLE_SIZE);
	}

	for (int i = 0; i < STRAND_SIZE; i++) {
		for (int j = 0; j < PARTICLE_SIZE; j++) {
			double r = j / PARTICLE_SIZE * 2 < 1 ? j / PARTICLE_SIZE * 2 : 1;
			double t = j * 0.3;
			double x = cos(t) * r;
			double y = t * 0.2;
			double z = sin(t) * r;

			s[i].p_p[j].x = x;
			s[i].p_p[j].y = -y;
			s[i].p_p[j].z = z + (i / PARTICLE_SIZE) * 10;

			s[i].r_p_p[j].x = x;
			s[i].r_p_p[j].y = -y;
			s[i].r_p_p[j].z = z + (i / PARTICLE_SIZE) * 10;
		}
	}
	for (int i = 0; i < STRAND_SIZE; i++) {
		double sum = 0;
		for (int j = 0; j < PARTICLE_SIZE; j++) {
			double3 edge = vector_sub(s[i].r_p_p[j], s[i].r_p_p[j]);
			sum += edge.norm();
		}
		sum /= (PARTICLE_SIZE - 1);
		cout << "rest_length : ";
		cout << sum << endl;
		s[i].r_p_l = sum;
	}
}

void helix_function() {
	
	for (double i = 0; i < p->pos.size(); i++) {
		for (double j = 0; j < p->pos[i].size(); j++) {
			int size = particle->pos[i].size();
			//radius 조절
			double r = j / size * 2 < 1 ? j / size * 2 : 1;
			//double r = j / size * 2 < 1 ? j / size : 1 - j / size;

			double t = j * 0.3;
			double x = cos(t) * r;
			double y = t * 0.2;
			double z = sin(t) * r;

			//helix hair
			p->pos[i][j] = Vector3d(x, -y, z + (i / particle->pos.size()) * 10);

			//bridge hair
			//p->pos[i][j] = Vector3d(z + (i / particle->pos.size()) * 10, x, -y);

			//p->pos[i][j] = Vector3d(x,-y,z + (i / p->pos.size()) * 10);
			//p->pos[i][j] = Vector3d(x,-y,z + (i / p->pos.size()));

			//straight hair
			//p->pos[i][j] = Vector3d(0, -j / p->pos.size() * 32, i / p->pos.size() * 32);

			//p->pos[i][j] = Vector3d(0.1*x,0.1*-y,0.1*z + (2.0 * i / p->pos.size()));

			p->velocity[i][j].setZero();
		}
	}
}

double3*  smoothing_function(double3 *lambda, double l, double alpha, bool is_position) {
	double beta = 0.0;

	double3  d[PARTICLE_SIZE];
	double3 pos[PARTICLE_SIZE];
	//lambda가 파티클 위치일 경우 return하기위한 pos vector

	for (int i = 0; i < PARTICLE_SIZE; i++) {
		d[i].x = lambda[i].x;
		d[i].y = lambda[i].y;
		d[i].z = lambda[i].z;
	}

	

	//beta formulation, l = 파티클간의 평균길이
	beta = 1 > 1 - exp(-1 / alpha) ? 1 - exp(-1 / alpha) : 1;

	d[0] = vector_sub(lambda[1], lambda[0]);
	for (int j = 1; j < PARTICLE_SIZE-1; j++) {
		int index_1 = j - 1 >= 0 ? j - 1 : 0;
		int index_2 = j - 2 >= 0 ? j - 2 : 0;
		auto term1 = vector_multiply(d[index_1], 2 * (1 - beta));
		auto term2 = vector_multiply(d[index_2], ((1 - beta) * (1 - beta)));
		auto term3 = vector_sub(term1,term2);
		auto term4 = vector_multiply(vector_sub(lambda[j + 1], lambda[j]), (beta * beta));
		d[j] =  vector_add(term3,term4);
	}
	

	if (is_position) {
		pos[0] = lambda[0];
		for (int j = 1; j < PARTICLE_SIZE; j++) {
			pos[j] = vector_add(d[j - 1], pos[j - 1]);
		}
	return pos;
	}
	
	return d;
}