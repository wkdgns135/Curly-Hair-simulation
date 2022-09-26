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
	vector<vector<Matrix3f>>	frames;
	vector<vector<Vector3f>>	t;
	vector<vector<float>>		m;
	vector<vector<float>>		w_c;
	vector<vector<float>>		w_d;
	vector<float>				rest_length;
};

void resize(vector<vector<Vector3f>> &v, vector<int> size);
void resize(vector<vector<Matrix3f>> &v, vector<int> size);
void resize(vector<vector<float>> &v, vector<int> size);

void compute_frame(Particle *p, vector<vector<Vector3f>> rest_p);
void multiply_vector(Vector3f &v1, Vector3f &v2, Vector3f &dest);
//Matrix3f rot_mat_from_two_vectors(Vec3<float> a, Vec3<float> b);
//Vec3<float> convert_vector(Matrix3f m, Vec3<float> v);
//Vec3<float> multiply_reference_vector_init(Matrix3f f, Vector3f e);
//Vec3<float> multiply_reference_vector(Matrix3f f, Vector3f e);