#pragma once
#include "HairModel.h"
#include <string>
#include <cstdio>
#include <iostream>

bool read_hair_asc(vector<vector<Vector3f>> &p, vector<int> &size, const char *filename) {
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

	nstrands = 10;

	for (int i = 0; i < nstrands; i++) {
		int nverts = 0;
		if (!fscanf(f, "%d", &nverts)) {
			fprintf(stderr, "Couldn't read number of vertices\n");
			fclose(f);
			return false;
		}
		vector<Vector3f> tmp;
		for (int j = 0; j < nverts; j++) {
			Vector3f v;
			if (!fscanf(f, "%f%f%f", &v[0],
				&v[1], &v[2])) {
				fprintf(stderr, "Couldn't read %d-th vertex in strand %d\n", j, i);
				fclose(f);
				return false;
			}
			if (nverts == 1 || nverts == 0)continue;
			v *= 100;
			v += Vector3f(0, -185, -25);
			tmp.push_back(v);
		}
		if (nverts == 1 || nverts == 0)continue;
		size.push_back(nverts);
		p.push_back(tmp);
	}

	fclose(f);
	return true;
}