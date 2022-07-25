#pragma once
#include "HairModel.h"
#include "vector_calc.h"

HairModel::HairModel() {
	for (int i = 0; i < STRAND_SIZE; i++) {
		s[i].p_p = (double3*)malloc(sizeof(double) * 3 * PARTICLE_SIZE); //double3 buffer size = 12
		s[i].s_p_p = (double3*)malloc(sizeof(double) * 3 * PARTICLE_SIZE);
		s[i].r_p_p = (double3*)malloc(sizeof(double) * 3 * PARTICLE_SIZE);
		s[i].r_s_p_p = (double3*)malloc(sizeof(double) * 3 * PARTICLE_SIZE);
		s[i].t = (double3*)malloc(sizeof(double) * 3 * PARTICLE_SIZE);
	}

	for (int i = 0; i < STRAND_SIZE; i++) {
		for (int j = 0; j < PARTICLE_SIZE; j++) {
			double r = j / double(PARTICLE_SIZE) * 2 < 1 ? j / double(PARTICLE_SIZE) * 2 : 1;
			double t = j * 0.3;
			double x = cos(t) * r;
			double y = t * 0.2;
			double z = sin(t) * r;

			s[i].p_p[j].x = x;
			s[i].p_p[j].y = -y;
			s[i].p_p[j].z = z + (i / double(STRAND_SIZE)) * 20;

			s[i].r_p_p[j].x = x;
			s[i].r_p_p[j].y = -y;
			s[i].r_p_p[j].z = z + (i / double(STRAND_SIZE)) * 20;
		}
	}

	for (int i = 0; i < STRAND_SIZE; i++) {
		double sum = 0;
		for (int j = 0; j < PARTICLE_SIZE; j++) {
			double3 edge = vector_sub(s[i].r_p_p[j], s[i].r_p_p[j]);
			sum += vector_length(edge);
		}
		sum /= (PARTICLE_SIZE - 1);
		cout << "rest_length : ";
		cout << sum << endl;
		s[i].r_p_l = sum;
	}

	for (int i = 0; i < STRAND_SIZE; i++) {
		s[i].r_s_p_p = smoothing_function(s[i].r_p_p, s[i].r_p_l, 0.23, true);
	}
}


void HairModel::draw_point() {
	glDisable(GL_LIGHTING);
	glPointSize(2.0f);
		glColor3f(0, 0, 0);

	for (int i = 0; i < STRAND_SIZE; i++) {
		for (int j = 0; j < PARTICLE_SIZE; j++) {
			glBegin(GL_POINTS);
			glVertex3f(s[i].p_p[j].x, s[i].p_p[j].y, s[i].p_p[j].z);
		}
	}
	glEnd();
	glPointSize(1.0f);
	glEnable(GL_LIGHTING);
}

void HairModel::draw_wire() {
	glDisable(GL_LIGHTING);
	glPointSize(2.0f);
	for (int i = 0; i < STRAND_SIZE; i++) {
		glBegin(GL_LINES);
		for (int j = 0; j < PARTICLE_SIZE-1; j++) {
			glVertex3f(s[i].p_p[j].x, s[i].p_p[j].y, s[i].p_p[j].z);
			glVertex3f(s[i].p_p[j+1].x, s[i].p_p[j+1].y, s[i].p_p[j+1].z);
		}
		glEnd();
	}
	glPointSize(1.0f);
	glEnable(GL_LIGHTING);
}

double3*  HairModel::smoothing_function(double3 *lambda, double l, double alpha, bool is_position) {
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
		double3 term1 = vector_multiply(d[index_1], 2 * (1 - beta));
		double3 term2 = vector_multiply(d[index_2], ((1 - beta) * (1 - beta)));
		double3 term3 = vector_sub(term1,term2);
		double3 term4 = vector_multiply(vector_sub(lambda[j + 1], lambda[j]), (beta * beta));
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