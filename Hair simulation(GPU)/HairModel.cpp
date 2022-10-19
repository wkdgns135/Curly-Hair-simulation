#pragma once
#include "HairModel.h"
#include "HairStyle.h"
#include "vector_calc.h"

HairModel::HairModel() {
	v = read_hair_asc("strand.txt");
	//open("rescaledCurlyHairs.txt"); // adjusting domain size

	sphere_pos = make_float3(0, -30, 0);
	sphere_radius = 10;

	STRAND_SIZE = v.size();
	for (int i = 0; i < v.size(); i++) {
		MAX_SIZE = MAX_SIZE < v[i].size() ? v[i].size() : MAX_SIZE;
		TOTAL_SIZE += v[i].size();
	}

	printf("TOTAL_SIZE : %d, MAX_SIZE : %d\n", TOTAL_SIZE, MAX_SIZE);

	particle_host.position = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	particle_host.velocity = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	particle_host.s_position = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	particle_host.r_position = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	particle_host.r_s_position = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	particle_host.s_velocity = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	particle_host.r_s_frame = (Frame*)malloc(sizeof(Frame) * TOTAL_SIZE);
	particle_host.s_frame = (Frame*)malloc(sizeof(Frame) * TOTAL_SIZE);
	particle_host.t = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	particle_host.d = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	particle_host.r_length = (double*)malloc(sizeof(double) * STRAND_SIZE);

	//Add color data
	particle_host.color = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	for (int i = 0; i < TOTAL_SIZE; i++)particle_host.color[i] = make_float3(1, 0, 0);

	vector2arr(v, particle_host.position);
	vector2arr(v, particle_host.r_position);

	for (int i = 0; i < v.size(); i++) {
		double sum = 0;
		for (int j = 0; j < v[i].size() - 1; j++) {
			float3 edge = vector_sub(v[i][j + 1], v[i][j]);
			sum += vector_length(edge);
		}

		sum /= (v[i].size() - 1);
		if (sum < 0.1) {
			cout << "rest_length : ";
			cout << sum << endl;
		}
		particle_host.r_length[i] = sum;
	}

	position_smoothing_function(particle_host.r_position, particle_host.r_s_position, particle_host.r_length, A_B, true);
	compute_frame(particle_host.r_s_frame, particle_host.r_s_position);

	int index = 0;
	for (int i = 0; i < v.size(); i++) {
		for (int j = 0; j < v[i].size(); j++) {
			if (v[i].size() - 1 == j) {
				index++; continue;
			}
			int index_1 = index - 1;
			int index0 = index;
			int index1 = index + 1;
			if (j == 0) {
				float3 e = vector_sub(particle_host.r_position[index1], particle_host.r_position[index0]);
				particle_host.t[index0] = multiply_transpose_frame(particle_host.r_s_frame[index0], e);
				particle_host.t[index0] = multiply_frame(particle_host.r_s_frame[index0], particle_host.t[index0]);
				index++;
				continue;
			}

			float3 e = vector_sub(particle_host.r_position[index1], particle_host.r_position[index0]);
			particle_host.t[index0] = multiply_transpose_frame(particle_host.r_s_frame[index_1], e);
			particle_host.t[index0] = multiply_frame(particle_host.r_s_frame[index_1], particle_host.t[index0]);
			index++;
		}
	}
	device_init();
	cout << MAX_SIZE << endl;

	// added by jhkim
	int gridSize = 256;
	int cellSize = gridSize * gridSize * gridSize;
	_hashing.init(TOTAL_SIZE, cellSize);
	//saveParticle("curlyHairs.txt");
}

void HairModel::open(char *filename)
{
	float3 minB = make_float3(10000.0, 10000.0, 10000.0);
	float3 maxB = make_float3(-10000.0, -10000.0, -10000.0);
	FILE *fp;

	fopen_s(&fp, filename, "r");
	int size = 0;
	float px, py, pz;
	fscanf(fp, "%d", &size);
	for (int i = 0; i < v.size(); i++) {
		for (int j = 0; j < v[i].size(); j++) {
			fscanf(fp, "%f %f %f", &px, &py, &pz);
			v[i][j].x = px;
			v[i][j].y = py;
			v[i][j].z = pz;
			minB.x = fminf(minB.x, px);
			minB.y = fminf(minB.y, py);
			minB.z = fminf(minB.z, pz);
			maxB.x = fmaxf(maxB.x, px);
			maxB.y = fmaxf(maxB.y, py);
			maxB.z = fmaxf(maxB.z, pz);
		}
	}
	fclose(fp);
	printf("min : %f, %f, %f\n", minB.x, minB.y, minB.z);
	printf("max : %f, %f, %f\n", maxB.x, maxB.y, maxB.z);
}

void HairModel::saveParticle(char *filename)
{
	FILE *fp;
	fopen_s(&fp, filename, "w");
	fprintf(fp, "%d\n", TOTAL_SIZE);
	for (int i = 0; i < v.size(); i++) {
		for (int j = 0; j < v[i].size(); j++) {
			fprintf(fp, "%f %f %f\n", v[i][j].x, v[i][j].y, v[i][j].z);
		}
	}
	fclose(fp);
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