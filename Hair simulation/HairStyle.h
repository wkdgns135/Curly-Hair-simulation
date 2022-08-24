#pragma once
#include "HairModel.h"
#include <string>

bool read_hair_asc(vector<vector<Vector3f>> p, const char *filename) {
	FILE *f = fopen(filename, "rb");
	if (!f) {
		fprintf(stderr, "Couldn't open %s\n", filename);
		return false;
	}

	int nstrands = 0;
	if (!fread(&nstrands, 4, 1, f)) {
		fprintf(stderr, "Couldn't read number of strands\n");
		fclose(f);
		return false;
	}

	for (int i = 0; i < nstrands; i++) {
		vector<Vector3f> p;
		int nverts = 0;
		
		if (!fread(&nverts, 4, 1, f)) {
			fprintf(stderr, "Couldn't read number of vertices\n");
			fclose(f);
			return false;
		}
		p[i].resize(nverts);

		for (int j = 0; j < nverts; j++) {
			if (!fread(&strands[i][j][0], 12, 1, f)) {
				fprintf(stderr, "Couldn't read %d-th vertex in strand %d\n", j, i);
				fclose(f);
				return false;
			}
		}
	}

	fclose(f);
	return true;

}