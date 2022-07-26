#pragma once
#include "vector_types.h"
#include <math.h>

void array_copy(double3 *a, double3 *b) {
	for (int i = 0; i < PARTICLE_SIZE; i++) {
		a[i].x = b[i].x;
		a[i].y = b[i].y;
		a[i].z = b[i].z;
	}
}

double3 vector_multiply(double3 a, double3 b) {
	double3 tmp;
	tmp.x = a.x * b.x;
	tmp.y = a.y * b.y;
	tmp.z = a.z * b.z;
	return tmp;
}


double3 vector_multiply(double3 a, double b) {
	double3 tmp;
	tmp.x = a.x * b;
	tmp.y = a.y * b;
	tmp.z = a.z * b;

	return tmp;
}

double3 vector_add(double3 a, double3 b) {
	double3 tmp;
	tmp.x = a.x + b.x;
	tmp.y = a.y + b.y;
	tmp.z = a.z + b.z;

	return tmp;
}


double3 vector_add(double3 a, double b) {
	double3 tmp;
	tmp.x = a.x + b;
	tmp.y = a.y + b;
	tmp.z = a.z + b;

	return tmp;
}

double3 vector_sub(double3 a, double3 b) {
	double3 tmp;
	tmp.x = a.x - b.x;
	tmp.y = a.y - b.y;
	tmp.z = a.z - b.z;

	return tmp;
}

double3 vector_sub(double3 a, double b) {
	double3 tmp;
	tmp.x = a.x - b;
	tmp.y = a.y - b;
	tmp.z = a.z - b;

	return tmp;
}

double vector_length(double3 a) {
	return sqrt(a.x*a.x + a.y * a.y + a.z * a.z);
}

double vector_normalize(double3 a) {
	double norm = vector_length(a);
	if (norm != 0) {
		a.x = a.x / norm;
		a.y = a.y / norm;
		a.z = a.z / norm;
	}
}

double3	vector_cross(double3 a, double3 b){
	double3 tmp;
	tmp.x = ((a.y*b.z) - (a.z*b.y));
	tmp.y = ((a.z*b.x) - (a.x*b.z));
	tmp.z = ((a.x*b.y) - (a.y*b.x));
	return tmp;
}

void compute_frame(Frame *f, double3 *p) {
	double3 aim = vector_sub(p[1], p[0]);
	vector_normalize(aim);

	double3 up;
	up.x = aim.z - aim.y;
	up.y = aim.x - aim.z;
	up.z = aim.y - aim.x;

	vector_normalize(up);
	for (int i = 1; i < PARTICLE_SIZE - 1; i++) {
		double3 aim = vector_sub(p[i+1], p[i]);
		vector_normalize(aim);

		double3 cross = vector_cross(aim, up);
		vector_normalize(cross);

		up = vector_cross(cross, aim);
		vector_normalize(up);

		f[i].aim.x = aim.x;
		f[i].aim.y = up.x;
		f[i].aim.z = cross.x;

		f[i].up.x = aim.y;
		f[i].up.y = up.y;
		f[i].up.z = cross.y;

		f[i].cross.x = aim.z;
		f[i].cross.y = up.z;
		f[i].cross.z = cross.z;

	}
}