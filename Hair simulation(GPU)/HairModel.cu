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
	//cudaMalloc((void**)&d,sizeof(float3) * TOTAL_SIZE);
	
	cudaMemcpy(p_p_d, p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_p_p_d, r_p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_s_p_p_d, r_s_p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_s_f_d, r_s_f, sizeof(Frame) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(r_s_p_p, h.r_s_p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(t_d, t, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_p_l_d, r_p_l, sizeof(double) * STRAND_SIZE, cudaMemcpyHostToDevice);


	array_init << <STRAND_SIZE, MAX_SIZE >> > (p_f_d);
	array_init << <STRAND_SIZE, MAX_SIZE >> > (p_v_d);
	//array_init << <STRAND_SIZE, MAX_SIZE >> > (d);

	//Init hash table
	hash_params.grid_size = make_uint3(128, 128, 128);
	hash_params.num_cells = (hash_params.grid_size.x * hash_params.grid_size.y * hash_params.grid_size.z);
	hash_params.world_origin = make_float3(-1.0f, -1.0f, -1.0f);
	hash_params.cell_size = make_float3(0.01, 0.01, 0.01);
	
	hashing.init(MAX_SIZE, &hash_params);
}

__global__ void collision_detect(float3 *p_p, float3 sphere, float radius, int x, int y) {
	if (threadIdx.x > y)return;
	if (blockIdx.x > x)return;
	if (blockIdx.x == 0)return;
	//if (threadIdx.x == 10)printf("%f %f %f\n", sphere.x, sphere.y, sphere.z);
	
	unsigned int tid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
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

	unsigned int tid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(p_f[tid], dt));
	p_f[tid] = make_float3(0.0, 0.0, 0.0);
	
} 

__global__ void integrate_internal_hair_force(float3 *p_p, float3 *r_p_p, float3 *s_p_p, float3 *r_s_p_p, Frame *s_f, float3* _t, float3 *p_f, float3 *p_v, int x, int y) {
	if (threadIdx.x > y - 2)return;
	//if (blockIdx.x > x)return;
	unsigned int tid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	//if(blockIdx.x == 0)printf("thread: %d\n", threadIdx.x);
	float3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
	float3 rest_e = vector_sub_k(r_p_p[tid + 1], r_p_p[tid]);
	float3 e_hat = vector_normalized_k(e);
	float3 b = vector_sub_k(s_p_p[tid + 1], s_p_p[tid]);
	float3 b_bar = vector_sub_k(r_s_p_p[tid + 1], r_s_p_p[tid]);
	float3 b_hat = vector_normalized_k(b);
	
	//float3 t;
	//if(threadIdx.x == 0) t = multiply_frame_k(s_f[tid], _t[tid]);
	//else t = multiply_frame_k(s_f[tid - 1], _t[tid]);
	
	float3 force1 = vector_multiply_k(e_hat,(vector_length_k(e)-vector_length_k(rest_e)) * K_S);
	float3 force2 = vector_multiply_k(vector_sub_k(e, rest_e), K_B);
	float3 force3_1 = vector_multiply_k(b_hat, vector_length_k(b) - vector_length_k(b_bar));
	float3 force3 = vector_multiply_k(force3_1, K_C);
	
	float3 result = vector_add_k(force1, force2);
	result = vector_add_k(result, force3);

	p_f[tid] = vector_add_k(p_f[tid], result);
	__syncthreads();
	p_f[tid + 1] = vector_sub_k(p_f[tid + 1], result);
	 
}

__global__ void integrate_external_hair_force(float3 *p_p, float3 *p_f, float3 *p_v, int x, int y) {
	if (threadIdx.x > y - 2)return;
	//if (blockIdx.x > x)return;
	unsigned int tid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	float3 gravity = make_float3(0.0, -10, 0.0);
	p_f[tid] = vector_add_k(p_f[tid], gravity);
}

__global__ void integrate_damping_force(float3 *p_p, float3 *s_p_p, float3 *s_p_v, float3 *p_f, float3 *p_v, int x, int y) {
	if (threadIdx.x > y - 2)return;
	//if (blockIdx.x > x)return;
	unsigned int tid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float3 d_v = vector_sub_k(p_v[tid + 1], p_v[tid]);
	float3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
	float3 e_hat = vector_normalized_k(e);

	float3 b = vector_sub_k(s_p_p[tid + 1], s_p_p[tid]);
	float3 b_hat = vector_normalized_k(b);
	float3 v = s_p_v[tid];

	float3 force1 = vector_multiply_k(vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat)), C_S);
	float3 force2 = vector_multiply_k(vector_sub_k(d_v, vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat))), C_B);
	float3 force3 = vector_multiply_k(b_hat, vector_dot_k(v, b_hat) * C_C);

	float3 result = vector_add_k(force1, force2);
	result = vector_add_k(result, force3);

	p_f[tid] = vector_add_k(p_f[tid], result);
	__syncthreads();
	p_f[tid + 1] = vector_sub_k(p_f[tid + 1], result);
	
}

__global__ void update_position(float3 *p_p, float3 *p_v, int x, int y) {
	//if (threadIdx.x > y)return;
	//if (blockIdx.x > x)return;
	if (threadIdx.x == 0)return;
	double dt = 0.00138883;
	unsigned int tid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	p_p[tid] = vector_add_k(p_p[tid], vector_multiply_k(p_v[tid], dt));
}
//float3 *lambda, double *l, double alpha, bool is_position

__global__ void position_smoothing_function_k(float3 *lambda, float3 *dst, float3 *d, double *l, float alpha) {
	unsigned int tid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	float beta = 0.0;

	beta = 1 > 1 - exp(-l[blockIdx.x] / alpha) ? 1 - exp(-l[blockIdx.x] / alpha) : 1;
	if (threadIdx.x == 0) {
		d[tid] = vector_sub_k(lambda[tid + 1], lambda[tid]);
	}
	else {
		int index_1 = threadIdx.x - 1 >= 0 ? tid - 1 : 0;
		int index_2 = threadIdx.x - 2 >= 0 ? tid - 2 : 0;
		float3 term1 = vector_multiply_k(d[index_1], 2 * (1 - beta));
		float3 term2 = vector_multiply_k(d[index_2], ((1 - beta) * (1 - beta)));
		float3 term3 = vector_sub_k(term1, term2);
		float3 term4 = vector_multiply_k(vector_sub_k(lambda[tid + 1], lambda[tid]), (beta * beta));
	}
	
	if (threadIdx.x == 0) {
		dst[tid] = lambda[tid];
	}
	else {
		dst[tid] = vector_add_k(d[tid - 1], dst[tid - 1]);
	}
}

__global__ void velocity_smoothing_function_k(float3 *lambda, float3 *dst, double *l, float alpha) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//if (threadIdx.x > 31) {
	//	__syncthreads();
	//	if (threadIdx.x > 63) {
	//		__syncthreads();
	//		if (threadIdx.x > 95) {
	//			__syncthreads();
	//		}
	//	}
	//}

	float beta = 1 > 1 - exp(-l[blockIdx.x] / alpha) ? 1 - exp(-l[blockIdx.x] / alpha) : 1;
	if (threadIdx.x == 0) {
		dst[tid] = vector_sub_k(lambda[tid + 1], lambda[tid]);
	}
	else {
		int index_1 = threadIdx.x - 1 >= 0 ? tid - 1 : 0;
		int index_2 = threadIdx.x - 2 >= 0 ? tid - 2 : 0;
		float3 term1 = vector_multiply_k(dst[index_1], 2 * (1 - beta));
		float3 term2 = vector_multiply_k(dst[index_2], ((1 - beta) * (1 - beta)));
		float3 term3 = vector_sub_k(term1, term2);
		float3 term4 = vector_multiply_k(vector_sub_k(lambda[tid + 1], lambda[tid]), (beta * beta));
		dst[tid] = vector_add_k(term3, term4);
	}
}

__global__ void move_root_up_k(float3 *p_p) {
	//if (threadIdx.x > y)return;
	//if (blockIdx.x > x)return;
	if (threadIdx.x != 0)return;
	unsigned int tid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	p_p[tid].y += 0.05;
}

__global__ void move_root_down_k(float3 *p_p) {
	//if (threadIdx.x > y)return;
	//if (blockIdx.x > x)return;
	if (threadIdx.x != 0)return;
	
	unsigned int tid = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	p_p[tid].y -= 0.05;
}

void HairModel::move_root(int dst) {
	if (dst == 0) {
		move_root_up_k << <STRAND_SIZE, MAX_SIZE >> > (p_p_d);
	}
	if (dst == 1) {
		move_root_down_k << <STRAND_SIZE, MAX_SIZE >> > (p_p_d);
	}
}

void HairModel:: simulation() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	//Calc hash table
	//updateHashTable(hashing, p_p_d, MAX_SIZE, STRAND_SIZE, MAX_SIZE, hash_params);

	for (int iter1 = 0; iter1 < 10; iter1++) {
		//collision_detect << <STRAND_SIZE, MAX_SIZE >> > (p_p_d, sphere_pos, sphere_radius, STRAND_SIZE, MAX_SIZE);
		position_smoothing_function(p_p, s_p_p, r_p_l, A_B, true);
		cudaMemcpy(s_p_p_d, s_p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
		//position_smoothing_function_k << <STRAND_SIZE, MAX_SIZE >> > (p_p_d, s_p_p_d, d,r_p_l_d, A_B);
		//compute_frame(s_f, s_p_p);
		//cudaMemcpy(s_f_d, s_f, sizeof(Frame) * TOTAL_SIZE, cudaMemcpyHostToDevice);
		
		for (int iter2 = 0; iter2 < 15; iter2++) {
			//velocity_smoothing_function_k << <STRAND_SIZE, MAX_SIZE >> > (p_v_d, s_p_v_d, r_p_l_d, A_C);
			cudaMemcpy(p_v, p_v_d, sizeof(float3) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
			velocity_smoothing_function(p_v, s_p_v, r_p_l, A_C, false);
			cudaMemcpy(s_p_v_d, s_p_v, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);

			integrate_internal_hair_force <<<STRAND_SIZE, MAX_SIZE>>> (p_p_d, r_p_p_d, s_p_p_d, r_s_p_p_d, r_s_f_d, t_d , p_f_d, p_v_d, STRAND_SIZE, MAX_SIZE);
			integrate_external_hair_force <<<STRAND_SIZE, MAX_SIZE>>> (p_p_d, p_f_d, p_v_d, STRAND_SIZE, MAX_SIZE);
			integrate << <STRAND_SIZE, MAX_SIZE >> > (p_p_d, p_f_d, p_v_d, 9.25887e-05, STRAND_SIZE, MAX_SIZE);

			for (int iter3 = 0; iter3 < 10; iter3++) {
				integrate_damping_force <<<STRAND_SIZE, MAX_SIZE >> > (p_p_d,s_p_p_d, s_p_v_d, p_f_d, p_v_d, STRAND_SIZE, MAX_SIZE);
				integrate << <STRAND_SIZE, MAX_SIZE >> > (p_p_d, p_f_d, p_v_d, 9.25887e-06, STRAND_SIZE, MAX_SIZE);
			}
		}
		update_position <<<STRAND_SIZE, MAX_SIZE >> > (p_p_d, p_v_d, STRAND_SIZE, MAX_SIZE);
	}


	//performace check

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << " SAXPY execution time : " << milliseconds << " ms " << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(p_p, p_p_d, sizeof(float3) * TOTAL_SIZE, cudaMemcpyDeviceToHost);	
}

