#ifndef __VECTOR_CALC__
#define __VECTOR_CALC__

#include "vector_types.h"
#include <math.h>

void array_copy(float3 *a, float3 *b) {
	for (int i = 0; i < PARTICLE_SIZE; i++) {
		a[i].x = b[i].x;
		a[i].y = b[i].y;
		a[i].z = b[i].z;
	}
}

float3 vector_multiply(float3 a, float3 b) {
	float3 tmp;
	tmp.x = a.x * b.x;
	tmp.y = a.y * b.y;
	tmp.z = a.z * b.z;
	return tmp;
}

float3 vector_multiply(float3 a, double b) {
	float3 tmp;
	tmp.x = a.x * b;
	tmp.y = a.y * b;
	tmp.z = a.z * b;

	return tmp;
}

float3 vector_add(float3 a, float3 b) {
	float3 tmp;
	tmp.x = a.x + b.x;
	tmp.y = a.y + b.y;
	tmp.z = a.z + b.z;

	return tmp;
}


float3 vector_add(float3 a, double b) {
	float3 tmp;
	tmp.x = a.x + b;
	tmp.y = a.y + b;
	tmp.z = a.z + b;

	return tmp;
}

float3 vector_sub(float3 a, float3 b) {
	float3 tmp;
	tmp.x = a.x - b.x;
	tmp.y = a.y - b.y;
	tmp.z = a.z - b.z;

	return tmp;
}

float3 vector_sub(float3 a, double b) {
	float3 tmp;
	tmp.x = a.x - b;
	tmp.y = a.y - b;
	tmp.z = a.z - b;

	return tmp;
}

double vector_length(float3 a) {
	return sqrt(a.x*a.x + a.y * a.y + a.z * a.z);
}

void vector_normalize(float3 &a) {
	double norm = vector_length(a);
	if (norm != 0) {
		a.x = a.x / norm;
		a.y = a.y / norm;
		a.z = a.z / norm;
	}
}

float3 vector_normalized(float3 a) {
	double norm = vector_length(a);
	if (norm != 0) {
		a.x = a.x / norm;
		a.y = a.y / norm;
		a.z = a.z / norm;
	}
	return a;
}


float3	vector_cross(float3 a, float3 b){
	float3 tmp;
	tmp.x = ((a.y*b.z) - (a.z*b.y));
	tmp.y = ((a.z*b.x) - (a.x*b.z));
	tmp.z = ((a.x*b.y) - (a.y*b.x));
	return tmp;
}

void compute_frame(Frame *f, float3 *p) {
	for (int i = 0; i < STRAND_SIZE; i++) {
		float3 aim = vector_sub(p[i * PARTICLE_SIZE + 1], p[i * PARTICLE_SIZE]);
		vector_normalize(aim);

		float3 up;
		up.x = aim.z - aim.y;
		up.y = aim.x - aim.z;
		up.z = aim.y - aim.x;
		vector_normalize(up);

		for (int j = 1; j < PARTICLE_SIZE - 1; j++) {
			int index = i * PARTICLE_SIZE + j;
			float3 aim = vector_sub(p[index + 1], p[index]);
			vector_normalize(aim);

			float3 cross = vector_cross(aim, up);
			vector_normalize(cross);

			up = vector_cross(cross, aim);
			vector_normalize(up);

			f[index].aim.x = aim.x;
			f[index].aim.y = up.x;
			f[index].aim.z = cross.x;

			f[index].up.x = aim.y;
			f[index].up.y = up.y;
			f[index].up.z = cross.y;

			f[index].cross.x = aim.z;
			f[index].cross.y = up.z;
			f[index].cross.z = cross.z;
		}
	}
}

float3 multiply_transpose_frame(Frame f, float3 e) {
	float3 tmp;
	tmp.x =
		e.x * f.aim.x +
		e.y * f.up.x +
		e.z * f.cross.x;
	
	tmp.y =
		e.x * f.aim.y +
		e.y * f.up.y +
		e.z * f.cross.y;

	tmp.z =
		e.x * f.aim.z +
		e.y * f.up.z +
		e.z * f.cross.z;
	return tmp;
}

float3 multiply_frame(Frame f, float3 e) {
	float3 tmp;
	tmp.x =
		e.x * f.aim.x +
		e.y * f.aim.y +
		e.z * f.aim.z;

	tmp.y =
		e.x * f.up.x +
		e.y * f.up.y +
		e.z * f.up.z;

	tmp.z =
		e.x * f.cross.x +
		e.y * f.cross.y +
		e.z * f.cross.z;
	return tmp;
}

float3* smoothing_function(float3 *lambda, double *l, double alpha, bool is_position) {
	double beta = 0.0;

	float3  *d = new float3[STRAND_SIZE * PARTICLE_SIZE];
	float3 *pos = new float3[STRAND_SIZE * PARTICLE_SIZE];
	//lambda가 파티클 위치일 경우 return하기위한 pos vector

	array_copy(d, lambda);

	//beta formulation, l = 파티클간의 평균길이

	for (int i = 0; i < STRAND_SIZE; i++) {
		int index = i * PARTICLE_SIZE;
		d[index] = vector_sub(lambda[index + 1], lambda[index]);
		beta = 1 > 1 - exp(-l[i] / alpha) ? 1 - exp(-l[i] / alpha) : 1;
		for (int j = 1; j < PARTICLE_SIZE - 1; j++) {
			int index_1 = j - 1 >= 0 ? j - 1 : 0;
			int index_2 = j - 2 >= 0 ? j - 2 : 0;

			int index1 = i * PARTICLE_SIZE + index_1;
			int index2 = i * PARTICLE_SIZE + index_2;
			index = i * PARTICLE_SIZE + j;

			float3 term1 = vector_multiply(d[index_1], 2 * (1 - beta));
			float3 term2 = vector_multiply(d[index_2], ((1 - beta) * (1 - beta)));
			float3 term3 = vector_sub(term1, term2);
			float3 term4 = vector_multiply(vector_sub(lambda[index + 1], lambda[index]), (beta * beta));
			d[index] = vector_add(term3, term4);
		}
	}

	if (is_position) {
		for (int i = 0; i < STRAND_SIZE; i++) {
			int index = i * PARTICLE_SIZE;
			pos[index] = lambda[index];
			for (int j = 1; j < PARTICLE_SIZE; j++) {
				index = i * PARTICLE_SIZE + j;
				pos[index] = vector_add(d[index - 1], pos[index - 1]);
			}

		}
		return pos;
	}
	return d;
}
#endif