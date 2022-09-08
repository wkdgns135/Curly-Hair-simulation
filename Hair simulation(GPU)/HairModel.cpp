#pragma once
#include "HairModel.h"
#include "HairStyle.h"
#include "vector_calc.h"

HairModel::HairModel() {
	v = read_hair_asc("strand.txt");

	sphere_pos = make_float3(0, -15, 0);
	sphere_radius = 10;

	STRAND_SIZE = v.size();
	for (int i = 0; i < v.size(); i++) {
		MAX_SIZE = MAX_SIZE < v[i].size() ? v[i].size() : MAX_SIZE;
		TOTAL_SIZE += v[i].size();
	}
	p_i = (int*)malloc(sizeof(int) * TOTAL_SIZE);
	p_p = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	p_v = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	s_p_p = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	r_p_p = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	r_s_p_p = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	p_v_d = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	s_p_v = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	r_s_f = (Frame*)malloc(sizeof(Frame) * TOTAL_SIZE);
	s_f = (Frame*)malloc(sizeof(Frame) * TOTAL_SIZE);
	t = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	r_p_l = (double*)malloc(sizeof(double) * STRAND_SIZE);

	vector2arr(v, p_p);
	vector2arr(v, r_p_p);

	for (int i = 0; i < v.size(); i++) {
		double sum = 0;
		for (int j = 0; j < v[i].size() - 1; j++) {
			float3 edge = vector_sub(v[i][j + 1], v[i][j]);
			sum += vector_length(edge);
		}

		sum /= (v[i].size() - 1);
		//cout << "rest_length : ";
		//cout << sum << endl;
		r_p_l[i] = sum;
	}

	array_copy(r_s_p_p, smoothing_function(r_p_p, r_p_l, 0.23, true));
	compute_frame(r_s_f, r_s_p_p);
	
	int index = 0;
	for(int i = 0; i < v.size(); i++){
		for (int j = 0; j < v[i].size(); j++) {
			if (j == 0 || v[i].size() - 1 == j) {
				index++; 
				continue;
			}
			int index_1 = index - 1;
			int index0 = index;
			int index1 = index + 1;

			float3 e = vector_sub(r_p_p[index1], r_p_p[index0]);
			t[index0] = multiply_transpose_frame(r_s_f[index_1], e);
			index++;
		}
	}
	device_init();
}

void HairModel::move_sphere(float3 dst) {
	sphere_pos = vector_add(sphere_pos, dst);
}


//
//void HairModel::draw_point() {
//	glDisable(GL_LIGHTING);
//	glPointSize(2.0f);
//	glColor3f(0, 0, 0);
//
//	int index = 0;
//	for (int i = 0; i < v.size(); i++) {
//		for (int j = 0; j < v[i].size(); j++) {
//			glBegin(GL_POINTS);
//			glVertex3f(p_p[index].x, p_p[index].y, p_p[index].z);
//			index++;
//			//glVertex3f(s[i].r_s_p_p[j].x, s[i].r_s_p_p[j].y, s[i].r_s_p_p[j].z);
//		}
//	}
//	glEnd();
//	glPointSize(1.0f);
//	glEnable(GL_LIGHTING);
//}
//
//void HairModel::draw_wire() {
//	glDisable(GL_LIGHTING);
//	glPointSize(2.0f);
//	glColor3f(0, 0, 0);
//	
//	int index = 0;
//	for (int i = 0; i < v.size(); i++) {
//		glBegin(GL_LINES);
//		for (int j = 0; j < v[i].size(); j++) {
//			if (j == v[i].size() - 1) {
//				index++; 
//				continue;
//			}
//			int index0 = index;
//			int index1 = index + 1;
//			glVertex3f(p_p[index0].x, p_p[index0].y, p_p[index0].z);
//			glVertex3f(p_p[index1].x, p_p[index1].y, p_p[index1].z);
//			index++;
//			//glVertex3f(s[i].r_s_p_p[j].x, s[i].r_s_p_p[j].y, s[i].r_s_p_p[j].z);
//			//glVertex3f(s[i].r_s_p_p[j+1].x, s[i].r_s_p_p[j+1].y, s[i].r_s_p_p[j+1].z);
//		}
//		glEnd();
//	}
//	glPointSize(1.0f);
//	glEnable(GL_LIGHTING);
//}
//
//void HairModel::draw_frame() {
//	glDisable(GL_LIGHTING);
//	glPointSize(2.0f);
//
//	int index = 0;
//	for (int i = 0; i < v.size(); i++) {
//		for (int j = 0; j < v[i].size(); j++) {
//			glBegin(GL_LINES);
//			glColor3f(1, 0, 0);
//			glVertex3f(r_s_p_p[index].x,  r_s_p_p[index].y,  r_s_p_p[index].z);
//			glVertex3f(r_s_p_p[index].x + r_s_f[index].aim.x, r_s_p_p[index].y + r_s_f[index].aim.y, r_s_p_p[index].z + r_s_f[index].aim.z);
//
//			glVertex3f(r_s_p_p[index].x,  r_s_p_p[index].y,  r_s_p_p[index].z);
//			glVertex3f(r_s_p_p[index].x + r_s_f[index].up.x, r_s_p_p[index].y + r_s_f[index].up.y, r_s_p_p[index].z + r_s_f[index].up.z);
//
//			glVertex3f(r_s_p_p[index].x,  r_s_p_p[index].y,  r_s_p_p[index].z);
//			glVertex3f(r_s_p_p[index].x + r_s_f[index].cross.x, r_s_p_p[index].y + r_s_f[index].cross.y, r_s_p_p[index].z + r_s_f[index].cross.z);
//
//			index++;
//			glEnd();
//		}
//	}
//}