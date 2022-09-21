#pragma once
#include "Particle.h"

void resize(vector<vector<Vector3f>> &v, vector<int> size) {
	v.resize(size.size());
	for (int i = 0; i < v.size(); i++) {
		v[i].resize(size[i]);
		for (auto &j : v[i])j.setZero();
	}
}

void resize(vector<vector<Matrix3f>> &v, vector<int> size) {
	v.resize(size.size());
	for (int i = 0; i < v.size(); i++) {
		v[i].resize(size[i]);
	}
}

void resize(vector<vector<float>> &v, vector<int> size) {
	v.resize(size.size());
	for (int i = 0; i < v.size(); i++) {
		v[i].resize(size[i]);
	}
}

const Matrix3f cross_product_matrix(const Eigen::Vector3f &v)
{
	Matrix3f result;
	result <<	0, -v[2], v[1],
				v[2], 0, -v[0],
				-v[1], v[0], 0;
	return result;
}

Matrix3f rotation_matrix(const Vector3f &axisAngle)
{
	float theta = axisAngle.norm();
	Vector3f thetahat = axisAngle / theta;

	if (theta == 0)
		thetahat.setZero();

	Matrix3f result;
	result.setIdentity();
	result = cos(theta)*result + sin(theta)*cross_product_matrix(thetahat) + (1 - cos(theta))*thetahat*thetahat.transpose();
	return result;
}

void compute_frame(Particle *p) {
	for (int i = 0; i < p->frames.size(); i++) {
		Vector3f firstnormal = Vector3f(0, 1, 0);
		vector<Vector3f> tangents;
		vector<Vector3f> normals;
		for (int j = 0; j < p->frames[i].size() - 1; j++) {
			tangents.push_back((p->pos[i][j + 1] - p->pos[i][j]).normalize);
			normals.push_back(firstnormal);
		}
		tangents.push_back(tangents[p->frames.size() - 2]);
		normals.push_back(normals[p->frames.size() - 2]);

		for (int j = 0; j < p->frames[i].size() - 1; j++) {
			Vector3f bitangent = tangents[j].cross(tangents[j + 1]);

			if (bitangent.norm == 0) {
				normals[j + 1] = normals[j];
			}
			else {
				bitangent.normalized();
				float theta = acos(tangents[j].dot(tangents[j + 1]));
				Matrix3f rotmat;
				
				rotate(rotmat, theta, bitangent);
				normals[j + 1] = rotmat * normals[j];

			}
			Vector3f cross = normals[j].cross(tangents[j]);
			cross.normalize();

			p->frames[i][j] <<	
				aim.x(), up.x(), cross.x(),
				aim.y(), up.y(), cross.y(),
				aim.z(), up.z(), cross.z();	

		}
	}
	//for (int i = 0; i < p->frames.size(); i++) {
	//	Vector3f aim = p->pos[i][1] - p->pos[i][0];
	//	aim.normalize();
	//	Vector3f up(aim[2] - aim[1], aim[0] - aim[2], aim[1] - aim[0]);
	//	//Vector3f up(0, 1, 0);
	//	up.normalize();

	//	Vector3f cross = aim.cross(up);
	//	cross.normalize();

	//	p->frames[i][0] <<
	//		aim.x(), up.x(), cross.x(),
	//		aim.y(), up.y(), cross.y(),
	//		aim.z(), up.z(), cross.z();

	//	for (int j = 1; j < p->frames[i].size() - 1; j++) {
	//		Vector3f aim = p->pos[i][j + 1] - p->pos[i][j];
	//		aim.normalize();

	//		Vector3f cross = aim.cross(up);
	//		cross.normalize();

	//		up = cross.cross(aim);
	//		up.normalize();

	//		p->frames[i][j] <<	
	//			aim.x(), up.x(), cross.x(),
	//			aim.y(), up.y(), cross.y(),
	//			aim.z(), up.z(), cross.z();	

	//	}
	//}

}

void multiply_vector(Vector3f &v1, Vector3f &v2, Vector3f &dest) {
	Vector3f temp(
		v1.x()*v2.x(),
		v1.y()*v2.y(),
		v1.z()*v2.z()
	);
	dest = temp;
}
/*
Vec3<float> multiply_reference_vector_init(Frame f, Vec3<float> e) {
	Vec3<float> temp;
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


Vec3<float> multiply_reference_vector(Frame f, Vec3<float> e) {
	Vec3<float> temp;

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


Matrix3f rot_mat_from_two_vectors(Vec3<float> a, Vec3<float> b)
{
	a = a / a.GetNorm();
	float b_norm = b.GetNorm();
	b = b / b_norm;

	Vec3<float> v = a.Cross(b);
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

Vec3<float> convert_vector(Matrix3f m, Vec3<float> v) {
	Vector3f temp(v.x(), v.y(), v.z());
	temp = (m * temp);
	return Vec3<float>(temp.x(), temp.y(), temp.z());
}

*/