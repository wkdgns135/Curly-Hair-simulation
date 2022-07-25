#include "vector_types.h"

double3 vector_multiply(double3 a, double3 b) {
	double3 tmp;
	tmp.x = a.x * b.x;
	tmp.x = a.y * b.y;
	tmp.x = a.z * b.z;

	return tmp;
}


double3 vector_multiply(double3 a, double b) {
	double3 tmp;
	tmp.x = a.x * b;
	tmp.x = a.y * b;
	tmp.x = a.z * b;

	return tmp;
}

double3 vector_add(double3 a, double3 b) {
	double3 tmp;
	tmp.x = a.x + b.x;
	tmp.x = a.y + b.y;
	tmp.x = a.z + b.z;

	return tmp;
}


double3 vector_add(double3 a, double b) {
	double3 tmp;
	tmp.x = a.x + b;
	tmp.x = a.y + b;
	tmp.x = a.z + b;

	return tmp;
}

double3 vector_sub(double3 a, double3 b) {
	double3 tmp;
	tmp.x = a.x - b.x;
	tmp.x = a.y - b.y;
	tmp.x = a.z - b.z;

	return tmp;
}

double3 vector_sub(double3 a, double b) {
	double3 tmp;
	tmp.x = a.x - b;
	tmp.x = a.y - b;
	tmp.x = a.z - b;

	return tmp;
}







double sqrt(double input, double x)
{
	for (int i = 0; i < 10; i++) { x = (x + (input / x)) / 2; } 
	return x;
}

double3 vector_length()