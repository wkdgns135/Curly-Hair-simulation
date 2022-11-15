#pragma once
#include "HairModel.h"
#include "FileController.h"
#include "vector_calc.h"
#include "GL/glut.h"

HairModel::HairModel(char *filename, int num_strands) {
	hair_style = filename;
	v = read_hair_asc(hair_style, num_strands);
	//open("rescaledCurlyHairs.txt"); // adjusting domain size
	
	state = GENERAL_SIMULATION;

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
	particle_host.R = (float3*)malloc(sizeof(float3) * MAX_SIZE);
	particle_host.n_position = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	particle_host.density = (float*)malloc(sizeof(float) * TOTAL_SIZE);
	particle_host.saturation = (float*)malloc(sizeof(float) * TOTAL_SIZE);
	particle_host.test_pos = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);
	particle_host.wet_position = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);

	colors = (float3*)malloc(sizeof(float3) * TOTAL_SIZE);

	vector2arr(v, particle_host.position);
	vector2arr(v, particle_host.r_position);

	params_init();
	pre_compute();

	normalize_position();
	device_init();
	computeMaxDensity();

	//saveParticle("curlyHairs.txt");
}


HairModel::~HairModel() {
	free(particle_host.position);
	free(particle_host.velocity);
	free(particle_host.s_position);
	free(particle_host.r_position);
	free(particle_host.r_s_position);
	free(particle_host.s_velocity);
	free(particle_host.r_s_frame);
	free(particle_host.s_frame);
	free(particle_host.t);
	free(particle_host.d);
	free(particle_host.r_length);
	free(particle_host.R);
	free(particle_host.n_position);
	free(particle_host.density);
	free(particle_host.saturation);
	free(particle_host.wet_position);
	free(colors);

	device_free();
}


void HairModel::params_init() {
	params_host.K_S = 500000.0;
	params_host.C_S = 4000.0;
	
	params_host.K_B = 30000.0;
	params_host.C_B = 2000.0;
	
	params_host.K_C = 15000.0;
	params_host.C_C = 1000.0;
	
	params_host.A_B = 0.23;
	params_host.A_C = 1.0;

	params_host.R_C = 0;

	//

	// added by jhkim
	float res = 128.0f;
	params_host.grid_size = make_int3(res, res, res);
	params_host.num_cells = res * res * res;
	float3 dx = make_float3(1.0f / res, 1.0f / res, 1.0f / res);
	params_host.cell_size = dx;
	params_host.particle_radius = (float)(4.0 * 0.5 / params_host.grid_size.x);

	params_host.scaling = 0.7f;
	params_host.sphere_rad = 0.28f;
	params_host.sphere_pos = make_float3(0.5, -0.1, 0.5);
}

void HairModel::pre_compute() {
	for (int i = 0; i < v.size(); i++) {
		double sum = 0;
		for (int j = 0; j < v[i].size() - 1; j++) {
			float3 edge = v[i][j + 1] - v[i][j];
			sum += vector_length(edge);
		}

		sum /= (v[i].size() - 1);
		if (sum < 0.1) {
			cout << "rest_length : ";
			cout << sum << endl;
		}
		particle_host.r_length[i] = sum;
	}

	position_smoothing_function(particle_host.r_position, particle_host.r_s_position, particle_host.r_length, params_host.A_B, true);
	compute_frame(particle_host.r_s_frame, particle_host.r_s_position);


	for (float i = 0; i < MAX_SIZE; i++) {
		float r = i / MAX_SIZE * 2 < 1 ? i / MAX_SIZE * 2 : 1;

		float t = i * 0.3;
		float x = cos(t) * r;
		float y = t * 0.2;
		float z = sin(t) * r;

		//helix hair
		float3 pos = make_float3(x, -y, z + (i / MAX_SIZE) * 10);
		particle_host.R[int(i)] = pos;
	}


	//min, max boundary calculation
	min_b = make_float3(1000000.0, 1000000.0, 1000000.0);
	max_b = make_float3(-1000000.0, -1000000.0, -1000000.0);

	int index = 0;
	for (int i = 0; i < STRAND_SIZE; i++) {
		for (int j = 0; j < MAX_SIZE; j++) {
			min_b.x = (fmin(min_b.x, v[i][j].x));
			min_b.y = (fmin(min_b.y, v[i][j].y));
			min_b.z = (fmin(min_b.z, v[i][j].z));
			max_b.x = (fmax(max_b.x, v[i][j].x));
			max_b.y = (fmax(max_b.y, v[i][j].y));
			max_b.z = (fmax(max_b.z, v[i][j].z));

			particle_host.saturation[index] = (float)j / (float)(MAX_SIZE - 1);
			index++;
		}
	}

	index = 0;
	for (int i = 0; i < v.size(); i++) {
		for (int j = 0; j < v[i].size(); j++) {
			if (v[i].size() - 1 == j) {
				index++; continue;
			}
			int index_1 = index - 1;
			int index0 = index;
			int index1 = index + 1;
			if (j == 0) {
				float3 e = particle_host.r_position[index1] - particle_host.r_position[index0];
				particle_host.t[index0] = multiply_transpose_frame(particle_host.r_s_frame[index0], e);
				particle_host.t[index0] = multiply_frame(particle_host.r_s_frame[index0], particle_host.t[index0]);
				index++;
				continue;
			}

			float3 e = particle_host.r_position[index1] - particle_host.r_position[index0];
			particle_host.t[index0] = multiply_transpose_frame(particle_host.r_s_frame[index_1], e);
			particle_host.t[index0] = multiply_frame(particle_host.r_s_frame[index_1], particle_host.t[index0]);
			index++;
		}
	}
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
	sphere_pos = sphere_pos +  dst;
}



void HairModel::draw_point() {
	glDisable(GL_LIGHTING);
	glPointSize(2.0f);
	glColor3f(0, 0, 0);

	int index = 0;
	for (int i = 0; i < v.size(); i++) {
		for (int j = 0; j < v[i].size(); j++) {
			glBegin(GL_POINTS);
			glVertex3f(particle_host.position[index].x, particle_host.position[index].y, particle_host.position[index].z);
			index++;
			//glVertex3f(s[i].r_s_p_p[j].x, s[i].r_s_p_p[j].y, s[i].r_s_p_p[j].z);
		}
	}
	glEnd();
	glPointSize(1.0f);
	glEnable(GL_LIGHTING);
}

float3 SCALAR_TO_COLOR(float val)
{
	// T fColorMap[3][3] = {{0.960784314,0.498039216,0.011764706},{0,0,0},{0,0.462745098,0.88627451}};
	float fColorMap[5][3] = { { 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };   //Red->Blue
   //T fColorMap[5][3] = { { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 0, 1, 1 }, { 0, 0, 1 } };
   //T fColorMap[4][3] = {{0.15,0.35,1.0},{1.0,1.0,1.0},{1.0,0.5,0.0},{0.0,0.0,0.0}};
	float v = val;
	if (val > 1.0) v = 1.0; if (val < 0.0) v = 0.0; v *= 4.0;
	int low = (int)floor(v), high = (int)ceil(v);
	float t = v - low;
	float3 color;
	color.x = (fColorMap[low][0])*(1 - t) + (fColorMap[high][0])*t;
	color.y = (fColorMap[low][1])*(1 - t) + (fColorMap[high][1])*t;
	color.z = (fColorMap[low][2])*(1 - t) + (fColorMap[high][2])*t;
	return color;
}

void HairModel::draw_wire() {
	static double table[13][3] =
	{ { 0.4901960784313725, 0.6392156862745098, 0.9725490196078431 },
	{ 0.992156862745098, 0.5294117647058824, 0.8274509803921569 },
	{ 0.7176470588235294, 0.9450980392156863, 0.5411764705882353 },
	{ 0.9254901960784314, 0.8352941176470588, 0.5686274509803922 },
	{ 0.5372549019607843, 0.8235294117647059, 0.5450980392156863 },
	{ 0.5843137254901961, 0.7098039215686275, 0.6980392156862745 },
	{ 0.4705882352941176, 0.7490196078431373, 0.7254901960784314 },
	{ 0.5529411764705882, 0.8274509803921569, 0.7176470588235294 },
	{ 0.992156862745098, 0.9607843137254902, 0.4862745098039216 },
	{ 0.9333333333333333, 0.8509803921568627, 0.8392156862745098 },
	{ 0.7647058823529412, 0.6274509803921569, 0.603921568627451 },
	{ 0.5568627450980392, 0.5019607843137255, 0.4588235294117647 },
	{ 0.807843137254902, 0.3764705882352941, 0.4745098039215686 } };
	srand(123456);

		glPushMatrix();
		glDisable(GL_LIGHTING);
		glPointSize(2.0);
		//vec3 color(0, 0.462745098, 0.88627451);
		float3 color = make_float3(1.0, 0.8, 0.0);
		int index = 0;
		float3 l_pos = make_float3(1.0, 1.0, 1.0);
		float3 eyePos= make_float3(0.5, 1.0, 1.0);


		for (int i = 0; i < STRAND_SIZE; i++) {
			int id = 13 * ((double)rand() / RAND_MAX);
			int mode = id % 13;
			//color = make_float3(table[mode][0], table[mode][1], table[mode][2]);
			for (int j = 0; j < MAX_SIZE; j++) {
				if (j < MAX_SIZE - 1) {
					//float w = particle_host.density[index];
					//float w = particle_host.saturation[index];
					//float3 c = SCALAR_TO_COLOR(w);
					auto c = color;
					color = make_float3(c.x, c.y, c.z);
					float3 N = make_float3(particle_host.t[index].x, particle_host.t[index].y, particle_host.t[index].z);
					float3 pos = make_float3(particle_host.position[index].x, particle_host.position[index].y, particle_host.position[index].z);
					vector_normalize(N);
					float3 L = l_pos - pos;
					vector_normalize(L);
					// lambert 
					auto lambert = fabs(vector_dot(N,L));
					double shading = lambert;
					// phong
					float3 R = N * 2.0 - L;
					vector_normalize(R);
					float k_specular = 1.0f;
					float n_specular = 15.0f;
					float3 eyeVec = pos - eyePos;
					vector_normalize(eyeVec);
					float3 E = eyeVec * -1;
					vector_normalize(E);
					float specular = k_specular * powf(fabs(vector_dot(R,E)), n_specular);
					shading += specular;
					glColor3d(color.x * shading, color.y * shading, color.z * shading);
					glBegin(GL_LINES);
					glVertex3f(particle_host.position[index].x, particle_host.position[index].y, particle_host.position[index].z);
					glVertex3f(particle_host.position[index + 1].x, particle_host.position[index + 1].y, particle_host.position[index + 1].z);
					glEnd();
				}
				index++;
			}
		}
		glPointSize(1.0);
		glEnable(GL_LIGHTING);
		glPopMatrix();
}


void HairModel::get_colors() {
	static double table[13][3] =
	{ { 0.4901960784313725, 0.6392156862745098, 0.9725490196078431 },
	{ 0.992156862745098, 0.5294117647058824, 0.8274509803921569 },
	{ 0.7176470588235294, 0.9450980392156863, 0.5411764705882353 },
	{ 0.9254901960784314, 0.8352941176470588, 0.5686274509803922 },
	{ 0.5372549019607843, 0.8235294117647059, 0.5450980392156863 },
	{ 0.5843137254901961, 0.7098039215686275, 0.6980392156862745 },
	{ 0.4705882352941176, 0.7490196078431373, 0.7254901960784314 },
	{ 0.5529411764705882, 0.8274509803921569, 0.7176470588235294 },
	{ 0.992156862745098, 0.9607843137254902, 0.4862745098039216 },
	{ 0.9333333333333333, 0.8509803921568627, 0.8392156862745098 },
	{ 0.7647058823529412, 0.6274509803921569, 0.603921568627451 },
	{ 0.5568627450980392, 0.5019607843137255, 0.4588235294117647 },
	{ 0.807843137254902, 0.3764705882352941, 0.4745098039215686 } };
	srand(123456);

	//vec3 color(0, 0.462745098, 0.88627451);
	int index = 0;
	float3 l_pos = make_float3(1.0, 1.0, 1.0);
	float3 eyePos = make_float3(0.5, 1.0, 1.0);


	for (int i = 0; i < STRAND_SIZE; i++) {
		int id = 13 * ((double)rand() / RAND_MAX);
		int mode = id % 13;
		//color = make_float3(table[mode][0], table[mode][1], table[mode][2]);
		for (int j = 0; j < MAX_SIZE; j++) {
			if (j < MAX_SIZE - 1) {
				//float w = particle_host.density[index];
				//float w = particle_host.saturation[index];
				//float3 c = SCALAR_TO_COLOR(w);
				float3 N = make_float3(particle_host.t[index].x, particle_host.t[index].y, particle_host.t[index].z);
				float3 pos = make_float3(particle_host.position[index].x, particle_host.position[index].y, particle_host.position[index].z);
				vector_normalize(N);
				float3 L = l_pos - pos;
				vector_normalize(L);
				// lambert 
				auto lambert = fabs(vector_dot(N, L));
				double shading = lambert;
				// phong
				float3 R = N * 2.0 - L;
				vector_normalize(R);
				float k_specular = 1.0f;
				float n_specular = 15.0f;
				float3 eyeVec = pos - eyePos;
				vector_normalize(eyeVec);
				float3 E = eyeVec * -1;
				vector_normalize(E);
				float specular = k_specular * powf(fabs(vector_dot(R, E)), n_specular);
				shading += specular;
				colors[index] = make_float3(color.x * shading, color.y * shading, color.z * shading);
			}
			index++;
		}
	}
}


void HairModel::draw_vertex(float3 v) {
	glVertex3f(v.x, v.y, v.z);
}

void HairModel::test_draw() {
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glPointSize(2.0);
	//vec3 color(0, 0.462745098, 0.88627451);
	float3 color = make_float3(1.0, 0.8, 0.0);
	int index = 0;
	float3 l_pos = make_float3(1.0, 1.0, 1.0);
	float3 eyePos = make_float3(-1, 1, 0);

	for (int i = 0; i < STRAND_SIZE; i++) {
		for (int j = 0; j < MAX_SIZE; j++) {
			if (j < MAX_SIZE - 1) {
				//float w = particle_host.density[index];
				float w = particle_host.saturation[index];
				float3 c = SCALAR_TO_COLOR(w);
				//auto c = color;
				color = make_float3(c.x, c.y, c.z);
				float3 N = make_float3(particle_host.t[index].x, particle_host.t[index].y, particle_host.t[index].z);
				float3 pos = make_float3(particle_host.test_pos[index].x, particle_host.test_pos[index].y, particle_host.test_pos[index].z);
				vector_normalize(N);
				float3 L = l_pos - pos;
				vector_normalize(L);
				// lambert 
				auto lambert = fabs(vector_dot(N, L));
				double shading = lambert;
				// phong
				float3 R = N * 2.0 - L;
				vector_normalize(R);
				float k_specular = 20.0f;
				float n_specular = 300.0f;
				float3 eyeVec = pos - eyePos;
				vector_normalize(eyeVec);
				float3 E = eyeVec * -1;
				vector_normalize(E);
				float specular = k_specular * powf(fabs(vector_dot(R, E)), n_specular);
				shading += specular;
				glColor3d(color.x * shading, color.y * shading, color.z * shading);
				glBegin(GL_LINES);
				draw_vertex(particle_host.R[j]);
				draw_vertex(particle_host.R[j+1]);
				glEnd();
			}
			index++;
		}
	}
	glPointSize(1.0);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

void HairModel::normalize_position(void)
{
	float3 minB = min_b;
	float3 maxB = max_b;
	auto longAxis = fmax(fmax(fabs(maxB.x - minB.x), fabs(maxB.y - minB.y)), fabs(maxB.z - minB.z));
	auto cp = (maxB + minB) * 0.5;
	double scale = 0.7;
	int index = 0;

	for (int i = 0; i < STRAND_SIZE; i++) {
		for (int j = 0; j < MAX_SIZE; j++) {
			float3 pos = make_float3(particle_host.position[index].x, particle_host.position[index].y, particle_host.position[index].z);
			auto vec = pos - cp;
			vec = vec / longAxis;
			vec = vec * scale;
			pos = make_float3(0.5, 0.5, 0.5);
			pos = vec + pos;
			particle_host.n_position[index] = pos;
			index++;
		}
	}
}

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