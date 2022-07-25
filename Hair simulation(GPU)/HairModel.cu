#include "HairModel.cuh"
#include "HairModel.h"

void init() {
	
	double3 *p;
	helix_function();
}


void helix_function() {
	for (double i = 0; i < p->pos.size(); i++) {
		for (double j = 0; j < p->pos[i].size(); j++) {
			int size = particle->pos[i].size();
			//radius 조절
			double r = j / size * 2 < 1 ? j / size * 2 : 1;
			//double r = j / size * 2 < 1 ? j / size : 1 - j / size;

			double t = j * 0.3;
			double x = cos(t) * r;
			double y = t * 0.2;
			double z = sin(t) * r;

			//helix hair
			p->pos[i][j] = Vector3d(x, -y, z + (i / particle->pos.size()) * 10);

			//bridge hair
			//p->pos[i][j] = Vector3d(z + (i / particle->pos.size()) * 10, x, -y);

			//p->pos[i][j] = Vector3d(x,-y,z + (i / p->pos.size()) * 10);
			//p->pos[i][j] = Vector3d(x,-y,z + (i / p->pos.size()));

			//straight hair
			//p->pos[i][j] = Vector3d(0, -j / p->pos.size() * 32, i / p->pos.size() * 32);

			//p->pos[i][j] = Vector3d(0.1*x,0.1*-y,0.1*z + (2.0 * i / p->pos.size()));

			p->velocity[i][j].setZero();
		}
	}
}
void HairModel::pre_compute() {
	//rest 파티클간의 평균 길이 계산
	for (int i = 0; i < rest_particle->pos.size(); i++) {
		double sum = 0;
		for (int j = 0; j < rest_particle->pos[i].size() - 1; j++) {
			Vector3d edge = (rest_particle->pos[i][j + 1] - rest_particle->pos[i][j]);
			sum += edge.norm();
		}
		sum /= (rest_particle->pos[i].size() - 1);
		cout << "rest_length : ";
		cout << sum << endl;
		rest_particle->rest_length.push_back(sum);
	}

	//smoothed 된 rest 파티클 위치 저장
	smoothed_rest_particle->pos = smoothing_function(rest_particle->pos, rest_particle->rest_length, alpha_b, true);

	//smoothed curve frame pre-compute
	compute_frame(smoothed_rest_particle);
	for (int i = 0; i < smoothed_rest_particle->pos.size(); i++) {
		for (int j = 1; j < smoothed_rest_particle->pos[i].size() - 1; j++) {
			Vector3d e = rest_particle->pos[i][j + 1] - rest_particle->pos[i][j];

			smoothed_rest_particle->t[i][j] = smoothed_rest_particle->frames[i][j - 1].transpose() * e;
		}
	}
}