#pragma once
#include "HairModel.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h" 
#include "vector_calc.h"
#include <iostream>


void HairModel::device_init() {
	cudaMalloc((void**)&p_p_d,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	//cudaMalloc((void**)&p_v,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	//cudaMalloc((void**)&p_f,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	//cudaMalloc((void**)&r_s_p_p,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	//cudaMalloc((void**)&r_p_p,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	//cudaMalloc((void**)&r_s_f,sizeof(Frame) * STRAND_SIZE * PARTICLE_SIZE);
	//cudaMalloc((void**)&t,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	//cudaMalloc((void**)&r_p_l,sizeof(double) * STRAND_SIZE * PARTICLE_SIZE);
	
	cudaMemcpy(p_p_d, p_p, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(r_s_p_p, h.r_s_p_p, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(r_p_p, h.r_p_p, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(r_s_f, h.r_s_f, sizeof(Frame) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(t, h.t, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(r_p_l, h.r_p_l, sizeof(double) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
}

__global__ void test_k(double3 *p_p) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	p_p[tid].z += 0.0001;
}

__global__ void integrate_internal_hair_force(double3 *p_p, double3 *r_p_p, double3 *p_f) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x > 0) {
		if (threadIdx.x + 1< blockDim.x) {
			double3 e = vector_sub(p_p[tid + 1], p_p[tid]);
			double3 rest_e = vector_sub(r_p_p[tid + 1], r_p_p[tid]);
			double3 e_hat = vector_normalized(e);

			double3 force = vector_multiply(e_hat,(vector_length(e)-vector_length(rest_e)) * 5000);
			
			p_f[tid] = vector_add(p_f[tid], force);
			p_f[tid + 1] = vector_sub(p_f[tid + 1], force);
		}
	}


}

__global__ void simulation_k() {
	
}

void HairModel:: simulation() {
	for (int iter1 = 0; iter1 < 2; iter1++) {
		for (int iter2 = 0; iter2 < 15; iter2++) {
			test_k <<<STRAND_SIZE, PARTICLE_SIZE>>>(p_p_d);
			for (int iter3 = 0; iter2 < 10; iter2++) {
				//
			}
		}
	}
	cudaMemcpy(p_p, p_p_d, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyDeviceToHost);
}

