#pragma once
#include <stdio.h>
#include "HairModel.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h" 
#include "vector_calc.cuh"

void HairModel::device_info() {
	cudaDeviceProp  prop;

	int count;
	cudaGetDeviceCount(&count);

	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("   --- General Information for device %d ---\n", i);
		printf("Name:  %s\n", prop.name);
		printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
		printf("Clock rate:  %d\n", prop.clockRate);
		printf("Device copy overlap:  ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Kernel execution timeout :  ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("\n");

		printf("   --- Memory Information for device %d ---\n", i);
		printf("Total global mem:  %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem:  %ld\n", prop.totalConstMem);
		printf("Max mem pitch:  %ld\n", prop.memPitch);
		printf("Texture Alignment:  %ld\n", prop.textureAlignment);
		printf("\n");

		printf("   --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
		printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp:  %d\n", prop.regsPerBlock);
		printf("Threads in warp:  %d\n", prop.warpSize);
		printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");
	}
}

void HairModel::device_init() {
	device_info();

	cudaMalloc((void**)&p_p_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&p_v_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&p_f_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&r_p_p_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&s_p_p_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&s_p_v_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&r_s_p_p_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&s_f_d,sizeof(Frame) * TOTAL_SIZE);
	cudaMalloc((void**)&t_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&r_p_l_d,sizeof(double) * TOTAL_SIZE);
	
	cudaMemcpy(p_p_d, p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_p_p_d, r_p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_s_p_p_d, r_s_p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(r_s_f, h.r_s_f, sizeof(Frame) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(r_s_p_p, h.r_s_p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(t_d, t, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_p_l_d, r_p_l, sizeof(double) * STRAND_SIZE, cudaMemcpyHostToDevice);


	array_init << <STRAND_SIZE, MAX_SIZE >> > (p_f_d);
	array_init << <STRAND_SIZE, MAX_SIZE >> > (p_v_d);
}

__global__ void collision_detect(float3 *p_p, float3 sphere, float radius, int x, int y) {
	if (threadIdx.x > y)return;
	if (blockIdx.x > x)return;
	if (blockIdx.x == 0)return;
	//if (threadIdx.x == 10)printf("%f %f %f\n", sphere.x, sphere.y, sphere.z);
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float3 &p = p_p[tid];
	float3 dir = vector_sub_k(p, sphere);
	float dist = vector_length_k(dir);
	if (dist < (radius + 0.01f)) {
		vector_normalize_k(dir);
		float3 new_pos = vector_add_k(sphere,vector_multiply_k(dir, (radius + 0.01f)));
		p_p[tid] = new_pos;
	}
}

__global__ void integrate(float3 *p_p, float3 *p_f, float3 *p_v, double dt, int x, int y) {
	//if (threadIdx.x > y)return; 
	//if (blockIdx.x > x)return;
	if (threadIdx.x == 0)return;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float3 ac = p_f[tid];
	p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
	p_f[tid] = make_float3(0.0, 0.0, 0.0);
	
}

__global__ void integrate_internal_hair_force(float3 *p_p, float3 *r_p_p, float3 *s_p_p, float3 *r_s_p_p, Frame *s_f, float3* _t, float3 *p_f, float3 *p_v, int x, int y) {
	if (threadIdx.x > y - 1)return;
	//if (blockIdx.x > x)return;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//if(blockIdx.x == 0)printf("thread: %d\n", threadIdx.x);
	float3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
	float3 rest_e = vector_sub_k(r_p_p[tid + 1], r_p_p[tid]);
	float3 e_hat = vector_normalized_k(e);

	float3 force1 = vector_multiply_k(e_hat,(vector_length_k(e)-vector_length_k(rest_e)) * K_S);

	float3 b = vector_sub_k(s_p_p[tid + 1], s_p_p[tid]);
	float3 b_bar = vector_sub_k(r_s_p_p[tid + 1], r_s_p_p[tid]);
	float3 b_hat = vector_normalized_k(b);
	
	//float3 force3 = vector_multiply_k(b_hat, K_C * (vector_length_k(b) - vector_length_k(b_bar)));


	//float3 t = multiply_frame_k(s_f[tid - 1], _t[tid]);
	//float3 force2 = vector_multiply_k(vector_sub_k(e, t), K_B);
	//float3 result = vector_add_k(force1, force2);

	p_f[tid] = vector_add_k(p_f[tid], force1);
	//p_f[tid] = vector_sub_k(p_f[tid], force3);
	__syncthreads();
	p_f[tid + 1] = vector_sub_k(p_f[tid + 1], force1);
	
}

__global__ void integrate_external_hair_force(float3 *p_p, float3 *p_f, float3 *p_v, int x, int y) {
	if (threadIdx.x > y - 1)return;
	//if (blockIdx.x > x)return;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	float3 gravity = make_float3(0.0, -20, 0.0);
	p_f[tid] = vector_add_k(p_f[tid], gravity);
}

__global__ void integrate_damping_force(float3 *p_p, float3 *s_p_p, float3 *s_p_v, float3 *p_f, float3 *p_v, int x, int y) {
	if (threadIdx.x > y - 1)return;
	//if (blockIdx.x > x)return;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	float3 d_v = vector_sub_k(p_v[tid + 1], p_v[tid]);
	float3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
	float3 e_hat = vector_normalized_k(e);

	float3 b = vector_sub_k(s_p_p[tid + 1], s_p_p[tid]);
	float3 b_hat = vector_normalized_k(b);
	float3 v = s_p_v[tid];

	float3 force1 = vector_multiply_k(vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat)), C_S);
	float3 force2 = vector_multiply_k(vector_sub_k(d_v, vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat))), C_B);
	//float3 force3 = vector_multiply_k(b_hat, vector_dot_k(v, b_hat) * C_C);

	float3 result = vector_add_k(force1, force2);
	p_f[tid] = vector_add_k(p_f[tid], result);
	//p_f[tid] = vector_sub_k(p_f[tid], force3);
	__syncthreads();
	p_f[tid + 1] = vector_sub_k(p_f[tid + 1], result);
	
}

__global__ void update_position(float3 *p_p, float3 *p_v, int x, int y) {
	//if (threadIdx.x > y)return;
	//if (blockIdx.x > x)return;
	if (threadIdx.x == 0)return;
	double dt = 0.01;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	p_p[tid] = vector_add_k(p_p[tid], vector_multiply_k(p_v[tid], dt));
	
}

void HairModel:: simulation() {
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start);

	collision_detect << <STRAND_SIZE, MAX_SIZE >> > (p_p_d, sphere_pos, sphere_radius, STRAND_SIZE, MAX_SIZE);
	for (int iter1 = 0; iter1 < 2; iter1++) {
		for (int iter2 = 0; iter2 < 15; iter2++) {
			//cudaMemcpy(p_p_d, s_p_p_d, sizeof(float3) * TOTAL_SIZE, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_v, p_v_d, sizeof(float3) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
			array_copy(s_p_p, smoothing_function(p_p, r_p_l, 0.23, true));
			array_copy(s_p_v, smoothing_function(p_v, r_p_l, 1.0, false));
			cudaMemcpy(s_p_p_d, s_p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
			cudaMemcpy(s_p_v_d, s_p_v, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);

			//cudaMemcpy(s_f_d, s_f, sizeof(Frame) * TOTAL_SIZE, cudaMemcpyHostToDevice);

			//compute_frame(s_f, s_p_p);
			
			integrate_internal_hair_force <<<STRAND_SIZE, MAX_SIZE >> > (p_p_d, r_p_p_d, s_p_p_d, r_s_p_p_d,s_f_d, t_d , p_f_d, p_v_d, STRAND_SIZE, MAX_SIZE);
			integrate_external_hair_force <<<STRAND_SIZE, MAX_SIZE >> > (p_p_d, p_f_d, p_v_d, STRAND_SIZE, MAX_SIZE);
			integrate << <STRAND_SIZE, MAX_SIZE >> > (p_p_d, p_f_d, p_v_d, 0.0009, STRAND_SIZE, MAX_SIZE);

			for (int iter3 = 0; iter3 < 10 * iter2; iter3++) {
				integrate_damping_force <<<STRAND_SIZE, MAX_SIZE >> > (p_p_d,s_p_p_d, s_p_v_d, p_f_d, p_v_d, STRAND_SIZE, MAX_SIZE);
				integrate << <STRAND_SIZE, MAX_SIZE >> > (p_p_d, p_f_d, p_v_d, 0.00009, STRAND_SIZE, MAX_SIZE);
			}
		}
		update_position <<<STRAND_SIZE, MAX_SIZE >> > (p_p_d, p_v_d, STRAND_SIZE, MAX_SIZE);
	}


	//performace check

	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float milliseconds = 0.0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//std::cout << " SAXPY execution time : " << milliseconds << " ms " << std::endl;

	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);

	cudaMemcpy(p_p, p_p_d, sizeof(float3) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
	
}

