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

Matrix3f rotation_matrix(float theta, Vector3f thetahat)
{
	if (theta < 0.0001)
		thetahat.setZero();

	Matrix3f result;
	result.setIdentity();
	result = cos(theta)*result + sin(theta)*cross_product_matrix(thetahat) + (1 - cos(theta))*thetahat*thetahat.transpose();
	return result;
}
int cnt = 0;
void compute_frame(Particle *p) {
	cnt++;
	for (int i = 0; i < p->frames.size(); i++) {
		Vector3f firsttangent(p->pos[i][1] - p->pos[i][0]);
		firsttangent.normalize();
		Vector3f firstnormal(0, 1, 0);
		Vector3f temp = firstnormal.cross(firsttangent);
		temp.normalize();
		firstnormal = firsttangent.cross(temp);
		firstnormal.normalize();

		vector<Vector3f> tangents(128);
		vector<Vector3f> normals(128);
		for (int j = 0; j < p->frames[i].size() - 1; j++) {
			Vector3f edge = p->pos[i][j + 1] - p->pos[i][j];
			edge.normalize();
			tangents[j] = (edge);
			normals[j] = (firstnormal);
		}
		tangents[127] = tangents[126];
		normals[127] = normals[126];
		
		for (int j = 0; j < p->frames[i].size() - 1; j++) {
			Vector3f bitangent = tangents[j].cross(tangents[j + 1]);

			if (bitangent.norm() == 0) {
				normals[j + 1] = normals[j];
			}
			else {
				bitangent.normalize();
				

				double theta = acos(double(tangents[j].dot(tangents[j + 1])));
				if (double(tangents[j].dot(tangents[j + 1])) >= 1)theta = 0;

				Matrix3f rotmat = rotation_matrix(theta, bitangent);
				
				//if (cnt > 43){
				//	cout << "theha:" <<theta << endl;
				//	cout << "bitan:" << bitangent << endl;
				//	cout << normals[j] << endl;
				//	cout << rotmat << endl;
				//}
				normals[j + 1] = rotmat * normals[j];
			}
			Vector3f cross = normals[j].cross(tangents[j]);
			cross.normalize();

			p->frames[i][j] <<	
				tangents[j].x(), normals[j].x(), cross.x(),
				tangents[j].y(), normals[j].y(), cross.y(),
				tangents[j].z(), normals[j].z(), cross.z();	

			//p->frames[i][j] <<
			//	tangents[j].x(), tangents[j].y(), tangents[j].z(),
			//	normals[j].x(), normals[j].y(), normals[j].z(),
			//	cross.x(), cross.y(), cross.z();
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

	//	//p->frames[i][0] <<
	//	//	aim.x(), up.x(), cross.x(),
	//	//	aim.y(), up.y(), cross.y(),
	//	//	aim.z(), up.z(), cross.z();

	//	p->frames[i][0] <<
	//		aim.x(), aim.y(), aim.z(),
	//		up.x(), up.y(), up.z(),
	//		cross.x(), cross.y(), cross.z();

	//	for (int j = 1; j < p->frames[i].size() - 1; j++) {
	//		Vector3f aim = p->pos[i][j + 1] - p->pos[i][j];
	//		aim.normalize();

	//		Vector3f cross = aim.cross(up);
	//		cross.normalize();

	//		up = cross.cross(aim);
	//		up.normalize();

	//		//p->frames[i][j] <<	
	//		//	aim.x(), up.x(), cross.x(),
	//		//	aim.y(), up.y(), cross.y(),
	//		//	aim.z(), up.z(), cross.z();	

	//		p->frames[i][j] <<
	//			aim.x(), aim.y(), aim.z(),
	//			up.x(), up.y(), up.z(),
	//			cross.x(), cross.y(), cross.z();

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