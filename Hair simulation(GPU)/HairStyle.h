#pragma once
#include "HairModel.h"
#include <string>
#include <cstdio>
#include <iostream>
#include "vector_calc.h"

bool read_hair_asc(float3 *p, vector<int> &size, const char *filename) {
	FILE *f = fopen(filename, "r");
	if (!f) {
		fprintf(stderr, "Couldn't open %s\n", filename);
		return false;
	}

	int nstrands = 0;
	if (!fscanf(f, "%d", &nstrands)) {
		fprintf(stderr, "Couldn't read number of strands\n");
		fclose(f);
		return false;
	}

	nstrands = 100;
	int index = 0;
	for (int i = 0; i < nstrands; i++) {
		int nverts = 0;
		if (!fscanf(f, "%d", &nverts)) {
			fprintf(stderr, "Couldn't read number of vertices\n");
			fclose(f);
			return false;
		}
		for (int j = 0; j < nverts; j++) {
			float f1, f2, f3;
			if (!fscanf(f, "%f%f%f", &p[index].x , &p[index].y, &p[index].z)) {
				fprintf(stderr, "Couldn't read %d-th vertex in strand %d\n", j, i);
				fclose(f);
				return false;
			}
			if (nverts == 1 || nverts == 0)continue;
			p[index] = vector_multiply(p[index], 100);
			p[index] = vector_add(p[index], make_float3(0, -185, -25));
			index++;
		}
	}

	fclose(f);
	return true;
}