#pragma once
#include "HairModel.h"
#include <string>
#include <cstdio>
#include <iostream>
#include <vector>
#include "vector_calc.h"
#include "vector_functions.h"

#include "GL/glut.h"

vector<vector<float3>> read_hair_asc(const char *filename) {

	vector<vector<float3>> tmp;
	//for (float i = 0; i < 1; i++) {
	//	vector<float3> v;
	//	for (float j = 0; j < 128; j++) {
	//		int size = 128;
	//		//radius Á¶Àý
	//		float r = j / size * 2 < 1 ? j / size * 2 : 1;
	//		//float r = j / size * 2 < 1 ? j / size : 1 - j / size;
	//
	//		float t = j * 0.3;
	//		float x = cos(t) * r;
	//		float y = t * 0.2;
	//		float z = sin(t) * r;
	//
	//		//helix hair
	//		float3 pos = make_float3(x, -y, z + (i / 128) * 10);
	//		pos = vector_add(pos, make_float3(0, 0, -100));
	//		v.push_back(pos);
	//	}
	//	tmp.push_back(v);
	//}

	FILE *f = fopen(filename, "r");
	if (!f) {
		fprintf(stderr, "Couldn't open %s\n", filename);
		return tmp;
	}
	
	int nstrands = 0;
	if (!fscanf(f, "%d", &nstrands)) {
		fprintf(stderr, "Couldn't read number of strands\n");
		fclose(f);
		return tmp;
	}
	
	nstrands = 64;
	for (int i = 0; i < nstrands; i++) {
		int nverts = 0;
		if (!fscanf(f, "%d", &nverts)) {
			fprintf(stderr, "Couldn't read number of vertices\n");
			fclose(f);
			return tmp;
		}
		vector<float3> tmp2;
		for (int j = 0; j < nverts; j++) {
			float3 tmp3;
			if (!fscanf(f, "%f%f%f", &tmp3.x , &tmp3.y, &tmp3.z)) {
				fprintf(stderr, "Couldn't read %d-th vertex in strand %d\n", j, i);
				fclose(f);
				return tmp;
			}
			if (nverts == 1 || nverts == 0)continue;
			tmp3 = vector_multiply(tmp3, 100);
			tmp3 = vector_add(tmp3, make_float3(0, -185, -25));
			tmp2.push_back(tmp3);
		}
		if (nverts == 1 || nverts == 0)continue;
		tmp.push_back(tmp2);
	}
	
	fclose(f);
	

	return tmp;
}

void vector2arr(vector<vector<float3>> v, float3 *p) {
	int index = 0;
	for (int i = 0; i < v.size(); i++) {
		for (int j = 0; j < v[i].size(); j++) {
			p[index++] = v[i][j];
		}
	}
}


void capture(int _width, int _height)
{
	static int _frame = 0;
	if (_frame == 0 || _frame % 2 == 0) {
		static int index = 0;
		char filename[100];
		sprintf_s(filename, "capture\\capture-%d.bmp", index);
		BITMAPFILEHEADER bf;
		BITMAPINFOHEADER bi;
		unsigned char *image = (unsigned char*)malloc(sizeof(unsigned char)*_width*_height * 3);
		FILE *file;
		fopen_s(&file, filename, "wb");
		if (image != NULL) {
			if (file != NULL) {
				glReadPixels(0, 0, _width, _height, 0x80E0, GL_UNSIGNED_BYTE, image);
				memset(&bf, 0, sizeof(bf));
				memset(&bi, 0, sizeof(bi));
				bf.bfType = 'MB';
				bf.bfSize = sizeof(bf) + sizeof(bi) + _width * _height * 3;
				bf.bfOffBits = sizeof(bf) + sizeof(bi);
				bi.biSize = sizeof(bi);
				bi.biWidth = _width;
				bi.biHeight = _height;
				bi.biPlanes = 1;
				bi.biBitCount = 24;
				bi.biSizeImage = _width * _height * 3;
				fwrite(&bf, sizeof(bf), 1, file);
				fwrite(&bi, sizeof(bi), 1, file);
				fwrite(image, sizeof(unsigned char), _height*_width * 3, file);
				fclose(file);
			}
			free(image);
		}
		//if (index == 60) { // cloth-bunny
		//if (index == 75) { // avatar
		//if (index == 122) { // rotating sphere
		//if (index == 213) { // rotating bunny
		if (index == 64) { // proximity_test0_2.obj
		//	exit(0);
		}
		index++;
	}
	_frame++;
}