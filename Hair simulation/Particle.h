#pragma once
#include "Vec3.h"
#include "GL/glut.h"
#include <vector>
#include <iostream>

#include "Sparse"
#include "Dense"
#include "svd"

using namespace std;
using namespace Eigen;

class Particle
{
public:
	vector<vector<Vector3f>>	pos;
	vector<vector<Vector3f>>	velocity;
	vector<vector<Vector3f>>	force;
	vector<vector<Matrix3d>>	frames;
	vector<vector<Vector3f>>	t;
	vector<vector<double>>		m;
	vector<vector<double>>		w_c;
	vector<vector<double>>		w_d;
	vector<double>				rest_length;
};

void resize(vector<vector<Vector3f>> &v, vector<int> size);
void resize(vector<vector<Matrix3d>> &v, vector<int> size);
void resize(vector<vector<double>> &v, vector<int> size);

void compute_frame(Particle *p);
void multiply_vector(Vector3f &v1, Vector3f &v2, Vector3f &dest);
//Matrix3f rot_mat_from_two_vectors(Vec3<double> a, Vec3<double> b);
//Vec3<double> convert_vector(Matrix3d m, Vec3<double> v);
//Vec3<double> multiply_reference_vector_init(Matrix3d f, Vector3f e);
//Vec3<double> multiply_reference_vector(Matrix3d f, Vector3f e);