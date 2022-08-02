#pragma once
#include "HairModel.cuh"

HairModel_d::HairModel_d(HairModel h) {
	cudaMalloc((void**)&p_p,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&p_v,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&p_f,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&r_s_p_p,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&r_p_p,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&r_s_f,sizeof(Frame) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&t,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&r_p_l,sizeof(double) * STRAND_SIZE * PARTICLE_SIZE);
	
	cudaMemcpy(p_p, h.p_p, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_s_p_p, h.r_s_p_p, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_p_p, h.r_p_p, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_s_f, h.r_s_f, sizeof(Frame) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(t, h.t, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_p_l, h.r_p_l, sizeof(double) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
}

__global__ void test_k(double3 *p_p, double3 *r_p_p) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < PARTICLE_SIZE; i++) {
		int index = tid + i;
		p_p[index].x += 0.0001;
	}
}

__global__ void simulation_k() {
	
}

double3* HairModel_d:: simulation() {
	for (int iter1 = 0; iter1 < 2; iter1++) {
		for (int iter2 = 0; iter2 < 15; iter2++) {
			test_k <<<1, STRAND_SIZE>>>(p_p);
			for (int iter3 = 0; iter2 < 10; iter2++) {
			}
		}
	}
}

