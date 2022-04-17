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

struct Frame {
public:
	Vec3<double> aim;
	Vec3<double> cross;
	Vec3<double> up;
	Vec3<double> pos;
	Vec3<double> t;
};

class Particle
{
public:
	double m = 1;
public:
	vector<vector<Vector3d>>	pos;
	vector<vector<Vector3d>>	velocity;
	vector<vector<Vector3d>>	force;
	vector<vector<Matrix3d>>	frames;
	vector<vector<Vector3d>>	t;
	vector<double>				rest_length;

public:
	double			inverss_mass(void) { return 1 / m; }
	//void			set_colvector(Vector3d, int, int);
};

void resize(vector<vector<Vector3d>> &v, vector<int> size);
void resize(vector<vector<Matrix3d>> &v, vector<int> size);

void compute_frame(Particle *p);
void multiply_vector(Vector3d &v1, Vector3d &v2, Vector3d &dest);
//Matrix3f rot_mat_from_two_vectors(Vec3<double> a, Vec3<double> b);
//Vec3<double> convert_vector(Matrix3d m, Vec3<double> v);
//Vec3<double> multiply_reference_vector_init(Matrix3d f, Vector3d e);
//Vec3<double> multiply_reference_vector(Matrix3d f, Vector3d e);