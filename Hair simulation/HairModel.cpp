#include "HairModel.h"

HairModel::HairModel() {
	particle = new Particle();
	smoothed_particle = new Particle();
	rest_particle = new Particle();
	smoothed_rest_particle = new Particle();
	
	for (int i = 0; i < 1; i++) {
		size.push_back(128);
	}

	resize(particle->pos, size);
	resize(particle->velocity, size);
	resize(particle->force, size);

	resize(smoothed_particle->pos, size);
	resize(smoothed_particle->velocity, size);
	resize(smoothed_particle->frames, size);
	resize(smoothed_particle->t, size);

	resize(rest_particle->pos, size);
	resize(rest_particle->velocity, size);

	resize(smoothed_rest_particle->pos, size);
	resize(smoothed_rest_particle->frames, size);
	resize(smoothed_rest_particle->t, size);

	init(particle);
	init(rest_particle);

	pre_compute();
}

void HairModel::init(Particle *p) {
	for (double i = 0; i < p->pos.size(); i++) {
		for (double j = 0; j < p->pos[i].size(); j++) {

			double t = j * 0.2;
			double x = cos(t);
			double y = t * 0.1;
			double z = sin(t);

			p->pos[i][j] = Vector3d(x,-y,z + (i / p->pos.size()) * 10);
			//p->pos[i][j] = Vector3d(x,-y,z + (i / p->pos.size()));
			//p->pos[i][j] = Vector3d(i / p->pos.size(), -j / p->pos.size(),0);
			//p->pos[i][j] = Vector3d(0.1*x,0.1*-y,0.1*z + (2.0 * i / p->pos.size()));
			p->velocity[i][j].setZero();
		}
	}
}

void HairModel::pre_compute() {
	//rest 파티클간의 평균 길이 계산
	for (int i = 0; i < rest_particle->pos.size(); i++) {
		double sum = 0;
		for (int j = 0; j < rest_particle->pos[i].size() - 1; j++) {
			Vector3d edge = (rest_particle->pos[i][j + 1] - rest_particle->pos[i][j]);
			sum += edge.norm();
		}
		sum /= (rest_particle->pos[i].size() - 1);
		cout << "rest_length : ";
		cout << sum << endl;
		rest_particle->rest_length.push_back(sum);
	}

	//smoothed 된 rest 파티클 위치 저장
	smoothed_rest_particle->pos = smoothing_function(rest_particle->pos, rest_particle->rest_length, alpha_b, true);
	
	//smoothed curve frame pre-compute
	compute_frame(smoothed_rest_particle);
	for (int i = 0; i < smoothed_rest_particle->pos.size(); i++) {
		//smoothed_rest_particle->t[i][0] = Vector3d(0, 0, 0);
		
		for (int j = 1; j < smoothed_rest_particle->pos[i].size() - 1; j++) {
			Vector3d e = rest_particle->pos[i][j + 1] - rest_particle->pos[i][j];
			
			smoothed_rest_particle->t[i][j] = smoothed_rest_particle->frames[i][j - 1].transpose() * e;
			//smoothed_rest_particle->frames[i][j].t = multiply_reference_vector_init(smoothed_rest_particle->frames[i][j-1], e);
			//smoothed_rest_particle->frames[i][j].t = e;
		}
	}
}

void HairModel::draw_point(vector<vector<Vector3d>> v) {
	glDisable(GL_LIGHTING);
	glPointSize(2.0f);
	glBegin(GL_POINTS);
	for (auto i : v)
		for(auto j : i)
		glVertex3f(j.x(), j.y(), j.z());
	glEnd();
	glPointSize(1.0f);
	glEnable(GL_LIGHTING);
}

void HairModel::draw_wire(vector<vector<Vector3d>> v) {
	glDisable(GL_LIGHTING);
	glPointSize(2.0f);
	
	for (auto i : v) {
		for (int j = 0; j < i.size()-1; j++) {
			glBegin(GL_LINES);
			glVertex3f(i[j].x(), i[j].y(), i[j].z());
			glVertex3f(i[j+1].x(), i[j+1].y(), i[j+1].z());
			glEnd();
		}
	}
	
	glPointSize(1.0f);
	glEnable(GL_LIGHTING);
}

void HairModel::draw_frame(Particle *p) {
	glDisable(GL_LIGHTING);
	glPointSize(2.0f);

	for (int i = 0; i < p->pos.size(); i++) {
		for (int j = 0; j < p->pos[i].size(); j++) {
			glBegin(GL_LINES);
			glColor3f(1, 0, 0);
			for (int k = 0; k < 3; k++) {
				glVertex3f(p->pos[i][j].x(), p->pos[i][j].y(), p->pos[i][j].z());
				glVertex3f(p->pos[i][j].x() + p->frames[i][j](k,0), p->pos[i][j].y() + p->frames[i][j](k, 1), p->pos[i][j].z() + p->frames[i][j](k, 2));
				
			}
			glBegin(GL_LINES);
			glColor3f(0, 1, 0);
			glVertex3f(p->pos[i][j].x(), p->pos[i][j].y(), p->pos[i][j].z());
			glVertex3f(p->pos[i][j].x() + p->t[i][j].x(), p->pos[i][j].y() + p->t[i][j].y(), p->pos[i][j].z() + p->t[i][j].z());

			glEnd();
		}
	}
}

void HairModel::simulation() {
	for (int iter1 = 0; iter1 < 2; iter1++) {
		for (int iter2 = 0; iter2 < 15; iter2++) {
			integrate_internal_hair_force();
			integrate_external_force();
			for (int iter3 = 0; iter3 < iter2 * 10; iter3++) {
				integrate_damping_force();
			}
		}
		update_position();
	}
}

void HairModel::stretch_spring_force(int i, int j) {
	if (j == particle->pos[i].size() - 1)return;

	Vector3d e = particle->pos[i][j+1] - particle->pos[i][j];
	Vector3d rest_e = rest_particle->pos[i][j+1] - rest_particle->pos[i][j];
	Vector3d e_hat = e;
	e_hat.normalize();

	Vector3d force = e_hat * (k_s * (e.norm() - rest_e.norm()));

	particle->force[i][j] += force;
	particle->force[i][j+1] -= force;
}

void HairModel::stretch_damping_force(int i, int j) {
	if (j == particle->pos[i].size() - 1)return;

	Vector3d delta_v = particle->velocity[i][j + 1] - particle->velocity[i][j];
	Vector3d e_hat = particle->pos[i][j + 1] - particle->pos[i][j];
	e_hat.normalize();

	Vector3d force = e_hat * ((delta_v.dot(e_hat)) * c_s);
	particle->force[i][j] += force;
	particle->force[i][j + 1] -= force;
}


void HairModel::bending_spring_force(int i, int j) {
	if (j == particle->pos[i].size() - 1)return;
	Vector3d t = smoothed_particle->frames[i][j - 1] * smoothed_rest_particle->t[i][j];
	smoothed_particle->t[i][j] = t;

	Vector3d e = particle->pos[i][j + 1] - particle->pos[i][j];
	Vector3d force = (e - t) * k_b;

	particle->force[i][j] += force;
	particle->force[i][j+1] -= force;
}


void HairModel::bending_damping_force(int i, int j) {
	if (j == particle->pos[i].size() - 1)return;

	Vector3d delta_v = particle->velocity[i][j + 1] - particle->velocity[i][j];
	Vector3d e_hat = particle->pos[i][j + 1] - particle->pos[i][j];
	e_hat.normalize();

	Vector3d force = (delta_v - (e_hat * (delta_v.dot(e_hat)))) * c_b;
	
	particle->force[i][j] += force;
	particle->force[i][j+1] -= force;
}

void HairModel::core_spring_force(int i, int j) {
	if (j == particle->pos[i].size() - 1)return;
	Vector3d b = smoothed_particle->pos[i][j + 1] - smoothed_particle->pos[i][j];
	Vector3d b_bar = smoothed_rest_particle->pos[i][j + 1] - smoothed_rest_particle->pos[i][j];
	Vector3d b_hat = b;
	b_hat.normalize();

	Vector3d force = b_hat * k_c * (b.norm() - b_bar.norm());
	b_hat = -b_hat;

	multiply_vector(b_hat, force, force);
	particle->force[i][j] += force;
	particle->force[i][j+1] -= force;
}

void HairModel::core_damping_force(int i, int j) {
	if (j == particle->pos[i].size() - 1)return;
	Vector3d b = smoothed_particle->pos[i][j + 1] - smoothed_particle->pos[i][j];
	Vector3d v = smoothed_particle->velocity[i][j];
	Vector3d b_hat = b;
	b_hat.normalize();

	Vector3d force = b_hat * c_c *(v.dot(b_hat));
	b_hat = -b_hat;
	multiply_vector(b_hat, force, force);
	particle->force[i][j] += force;
	particle->force[i][j + 1] -= force;
}

vector<vector<Vector3d>>  HairModel::smoothing_function(vector<vector<Vector3d>> lambda, vector<double> l,double alpha, bool is_position) {
	double beta = 0.0;

	vector<vector<Vector3d>>  d;
	vector<vector<Vector3d>> pos;
	//lambda가 파티클 위치일 경우 return하기위한 pos vector
	resize(d, size);
	resize(pos, size);

	copy(lambda.begin(), lambda.end(), d.begin());

	for (int i = 0; i < lambda.size(); i++) {
		//beta formulation, l = 파티클간의 평균길이
		beta = min(1.0, 1 - exp(-l[i] / alpha));

		d[i][0] = lambda[i][1] - lambda[i][0];
		for (int j = 1; j < lambda[i].size()-1; j++) {
			int index_1 = j - 1 >= 0 ? j - 1 : 0;
			int index_2 = j - 2 >= 0 ? j - 2 : 0;
			d[i][j] = d[i][index_1] * 2 * (1 - beta);
			d[i][j] -= d[i][index_2] * (1 - beta) * (1 - beta);
			d[i][j] += (lambda[i][j + 1] - lambda[i][j]) * beta * beta;
		}
	}

	if (is_position) {
		for (int i = 0; i < lambda.size(); i++) {
			pos[i][0] = lambda[i][0];
			for (int j = 1; j < lambda[i].size(); j++) {
				pos[i][j] = d[i][j - 1] + pos[i][j - 1];
			}
		}
		return pos;
	}
	return d;
}


void HairModel::integrate_internal_hair_force() {
	double dt = 9.25887e-05;
	smoothed_particle->pos = smoothing_function(particle->pos, rest_particle->rest_length, alpha_b, true);

	compute_frame(smoothed_particle);
	//spring forces 계산
	for (int i = 0; i < particle->pos.size(); i++) {
		for (int j = 0; j < particle->pos[i].size(); j++) {
			stretch_spring_force(i, j);
			bending_spring_force(i, j);
			core_spring_force(i, j);

			if (j == 0)continue;
			Vector3d acceleration = particle->force[i][j] * particle->inverss_mass();
			particle->velocity[i][j] += acceleration * dt;
			particle->force[i][j].setZero();
		}
	}
}

void HairModel::integrate_external_force() {
	double dt = 9.25887e-05;
	Vector3d gravity(0.0, -100.0, 0.0);
	for (int i = 0; i < particle->pos.size(); i++) {
		for (int j = 0; j < particle->pos[i].size(); j++) {
			particle->force[i][j] += gravity;

			if (j == 0)continue;
			Vector3d acceleration = particle->force[i][j] * particle->inverss_mass();
			particle->velocity[i][j] += acceleration * dt;
			particle->force[i][j].setZero();
		}
	}
}

void HairModel::integrate_damping_force() {
	double dt = 9.25887e-06;
	smoothed_particle->velocity = smoothing_function(particle->velocity, rest_particle->rest_length, alpha_c, false);
	for (int i = 0; i < particle->pos.size(); i++) {
		for (int j = 0; j < particle->pos[i].size(); j++) {
			stretch_damping_force(i, j);
			bending_damping_force(i, j);
			core_damping_force(i, j);

			if (j == 0)continue;
			Vector3d acceleration = particle->force[i][j] * particle->inverss_mass();
			particle->velocity[i][j] += acceleration * dt;
			particle->force[i][j].setZero();
		}
	}
}
void HairModel::update_position() {
	double dt = 0.00138883;
	for (int i = 0; i < particle->pos.size(); i++) {
		for (int j = 1; j < particle->pos[i].size(); j++) {
			particle->pos[i][j] += particle->velocity[i][j] * dt;

			//limit velocity
			//if (particle->velocity[i][j].norm() *  particle->velocity[i][j].norm() > v_threshold) {
			//	particle->velocity[i][j] /= particle->velocity[i][j].norm();
			//	particle->velocity[i][j] *= v_threshold;
			//}
		}
	}
}

void HairModel::move_root_particle(Vector3d dest) {
	for (auto &p : particle->pos) {
		p[0] += dest;
	}
}