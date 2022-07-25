#pragma once
#include "vector_types.h"
#include <math.h>

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