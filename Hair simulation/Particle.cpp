#pragma once
#include "Particle.h"

void resize(vector<vector<Vector3d>> &v, vector<int> size) {
	v.resize(size.size());
	for (int i = 0; i < v.size(); i++) {
		v[i].resize(size[i]);
		for (auto &j : v[i])j.setZero();
	}
}

void resize(vector<vector<Matrix3d>> &v, vector<int> size) {
	v.resize(size.size());
	for (int i = 0; i < v.size(); i++) {
		v[i].resize(size[i]);
	}
}

void compute_frame(Particle *p) {
	for (int i = 0; i < p->frames.size(); i++) {
		Vector3d up(0, 0, 1);
		up.normalize();
		for (int j = 0; j < p->frames[i].size() - 1; j++) {
			Vector3d aim = p->pos[i][j + 1] - p->pos[i][j];
			aim.normalize();

			Vector3d cross = aim.cross(up);
			cross.normalize();

			up = cross.cross(aim);
			up.normalize();

			p->frames[i][j] <<	aim.x(), up.x(), cross.x(),
								aim.y(), up.y(), cross.y(),
								aim.z(), up.z(), cross.z();	

		}
	}
}

void multiply_vector(Vector3d &v1, Vector3d &v2, Vector3d &dest) {
	Vector3d temp(
		v1.x()*v2.x(),
		v1.y()*v2.y(),
		v1.z()*v2.z()
	);
	dest = temp;
}
/*
Vec3<double> multiply_reference_vector_init(Frame f, Vec3<double> e) {
	Vec3<double> temp;
	temp.SetX(
		e.x() * f.aim.x() +
		e.y() * f.up.x() +
		e.z() * f.cross.x()
	);
	
	temp.SetY(
		e.x() * f.aim.y() +
		e.y() * f.up.y() +
		e.z() * f.cross.y()
	);
	
	temp.SetZ(
		e.x() * f.aim.z() +
		e.y() * f.up.z() +
		e.z() * f.cross.z()
	);
	return temp;
}


Vec3<double> multiply_reference_vector(Frame f, Vec3<double> e) {
	Vec3<double> temp;

	temp.SetX(
		e.x() * f.aim.x() +
		e.y() * f.aim.y() +
		e.z() * f.aim.z()
	);
	
	temp.SetY(
		e.x() * f.up.x() +
		e.y() * f.up.y() +
		e.z() * f.up.z()
	);
	
	temp.SetZ(
		e.x() * f.cross.x() +
		e.y() * f.cross.y() +
		e.z() * f.cross.z()
	);

	return temp;
}


Matrix3f rot_mat_from_two_vectors(Vec3<double> a, Vec3<double> b)
{
	a = a / a.GetNorm();
	float b_norm = b.GetNorm();
	b = b / b_norm;

	Vec3<double> v = a.Cross(b);
	float s = v.GetNorm();
	float c = a.Dot(b);
	Matrix3f vx; 
	vx.norm();
	vx << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
	Matrix3f r = Matrix3f::Identity(3, 3);
	if (s != 0) {
		r = r + vx + vx * vx*((1 - c) / pow(s, 2));
	}
	else {
		cout << "doesn't work if a == -b" << endl;
	}
	return r;
}

Vec3<double> convert_vector(Matrix3d m, Vec3<double> v) {
	Vector3f temp(v.x(), v.y(), v.z());
	temp = (m * temp);
	return Vec3<double>(temp.x(), temp.y(), temp.z());
}

*/