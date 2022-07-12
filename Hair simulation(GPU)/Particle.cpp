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

void resize(vector<vector<double>> &v, vector<int> size) {
	v.resize(size.size());
	for (int i = 0; i < v.size(); i++) {
		v[i].resize(size[i]);
	}
}

const Matrix3d cross_product_matrix(const Eigen::Vector3d &v)
{
	Matrix3d result;
	result <<	0, -v[2], v[1],
				v[2], 0, -v[0],
				-v[1], v[0], 0;
	return result;
}

Matrix3d rotation_matrix(const Vector3d &axisAngle)
{
	double theta = axisAngle.norm();
	Vector3d thetahat = axisAngle / theta;

	if (theta == 0)
		thetahat.setZero();

	Matrix3d result;
	result.setIdentity();
	result = cos(theta)*result + sin(theta)*cross_product_matrix(thetahat) + (1 - cos(theta))*thetahat*thetahat.transpose();
	return result;
}

void compute_frame(Particle *p) {

	//for (int i = 0; i < p->frames.size(); i++) {
	//	Vector3d t = p->pos[i][1] - p->pos[i][0];
	//	t.normalize();
	//	Vector3d u(t[2] - t[1], t[0] - t[2], t[1] - t[0]);
	//	u.normalize();
	//	Vector3d v = t.cross(u);
	//	v.normalize();

	//	Vector3d prev_t = t;
	//	Vector3d prev_u = u;

	//	/*
	//	p->frames[i][0] << 
	//		t.x(), t.y(), t.z(),
	//		u.x(), u.y(), u.z(),
	//		v.x(), v.y(), v.z();
	//	*/
	//	
	//	p->frames[i][0] <<	
	//		t.x(), u.x(), v.x(),
	//		t.y(), u.y(), v.y(),
	//		t.z(), u.z(), v.z();
	//	
	//	for (int j = 1; j < p->frames[i].size()-1; j++) {
	//		t = p->pos[i][j + 1] - p->pos[i][j];
	//		t.normalize();
	//		
	//		Vector3d n = prev_t.cross(t);
	//		Vector3d prev_t(p->frames[i][j - 1](0, 0), p->frames[i][j - 1](1, 0), p->frames[i][j - 1](2, 0));
	//		Vector3d prev_u(p->frames[i][j - 1](0, 1), p->frames[i][j - 1](1, 1), p->frames[i][j - 1](2, 1));


	//		if (n.norm() < 1e-10) {
	//			u = prev_u;
	//		}
	//		else {
	//			if (t.dot(prev_t) > 0) {
	//				u = rotation_matrix(n*asin(n.norm()) / n.norm()) * prev_u;
	//			}
	//			else {
	//				u = rotation_matrix(n*(M_PI - asin(n.norm())) / n.norm()) * prev_u;
	//			}
	//		}
	//		u.normalize();

	//		v = t.cross(u);
	//		//v.normalize();
	//		p->frames[i][j] <<
	//			t.x(), u.x(), v.x(),
	//			t.y(), u.y(), v.y(),
	//			t.z(), u.z(), v.z();
	//	}
	//}

	for (int i = 0; i < p->frames.size(); i++) {
		Vector3d aim = p->pos[i][1] - p->pos[i][0];
		aim.normalize();
		Vector3d up(aim[2] - aim[1], aim[0] - aim[2], aim[1] - aim[0]);
		up.normalize();

		for (int j = 1; j < p->frames[i].size() - 1; j++) {
			Vector3d aim = p->pos[i][j + 1] - p->pos[i][j];
			aim.normalize();

			Vector3d cross = aim.cross(up);
			cross.normalize();

			up = cross.cross(aim);
			up.normalize();

			p->frames[i][j] <<	
				aim.x(), up.x(), cross.x(),
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