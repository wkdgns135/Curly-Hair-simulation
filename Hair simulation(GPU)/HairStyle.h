#pragma once
#include "HairModel.h"
#include <string>
#include <cstdio>
#include <iostream>
#include "vector_calc.h"
#include "vector_functions.h"

vector<vector<float3>> read_hair_asc(const char *filename) {
	FILE *f = fopen(filename, "r");
	vector<vector<float3>> tmp;
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

	nstrands = 100;
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