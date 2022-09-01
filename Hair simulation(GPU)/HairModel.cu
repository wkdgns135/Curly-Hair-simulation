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

	cudaMalloc((void**)&p_i_d,sizeof(int) * TOTAL_SIZE);
	cudaMalloc((void**)&p_p_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&p_v_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&p_f_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&r_p_p_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&s_p_p_d,sizeof(float3) * TOTAL_SIZE);
	
	//cudaMalloc((void**)&r_s_p_p,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&s_f_d,sizeof(Frame) * TOTAL_SIZE);
	cudaMalloc((void**)&t_d,sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&r_p_l_d,sizeof(double) * TOTAL_SIZE);
	
	cudaMemcpy(p_i_d, p_i, sizeof(int) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(p_p_d, p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_p_p_d, r_p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(r_s_f, h.r_s_f, sizeof(Frame) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(r_s_p_p, h.r_s_p_p, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(t_d, t, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_p_l_d, r_p_l, sizeof(double) * STRAND_SIZE, cudaMemcpyHostToDevice);


	array_init << <STRAND_SIZE, MAX_SIZE >> > (p_f_d);
	array_init << <STRAND_SIZE, MAX_SIZE >> > (p_v_d);
}

//__global__ void integrate_internal_hair_force(int *p_i,float3 *p_p, float3 *r_p_p, Frame *s_f, float3* _t,float3 *p_f, float3 *p_v) {
//	double dt = 0.0009;
//	
//	for (int i = 0; i < PARTICLE_SIZE; i++) {
//		int tid = p_i[threadIdx.x * PARTICLE_SIZE + i];
//		if (i < PARTICLE_SIZE - 1) {
//			//Stretch spring
//			float3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
//			float3 rest_e = vector_sub_k(r_p_p[tid + 1], r_p_p[tid]);
//			float3 e_hat = vector_normalized_k(e);
//
//			float3 force = vector_multiply_k(e_hat, (vector_length_k(e) - vector_length_k(rest_e)) * K_S);
//
//			p_f[tid] = vector_add_k(p_f[tid], force);
//			p_f[tid + 1] = vector_sub_k(p_f[tid + 1], force);
//
//			//Bending spring
//			float3 t = multiply_frame_k(s_f[tid - 1], _t[tid]);
//			force = vector_multiply_k(vector_sub_k(e, t), K_B);
//			
//			p_f[tid] = vector_add_k(p_f[tid], force);
//			p_f[tid + 1] = vector_sub_k(p_f[tid + 1], force);
//			
//
//		}
//		if (i > 0) {
//			float3 ac = p_f[tid];
//			p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
//			p_f[tid] = make_float3(0.0, 0.0, 0.0);
//		}
//	}
//}

__global__ void integrate(float3 *p_p, float3 *p_f, float3 *p_v, double dt) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x > 0) {
		float3 ac = p_f[tid];
		p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
		p_f[tid] = make_float3(0.0, 0.0, 0.0);
	}
}

__global__ void integrate_internal_hair_force(float3 *p_p, float3 *r_p_p, Frame *s_f, float3* _t, float3 *p_f, float3 *p_v) {
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
	double dt = 0.0009;

	if (threadIdx.x < blockDim.x - 1) {
		//if(blockIdx.x == 0)printf("thread: %d\n", threadIdx.x);
		float3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
		float3 rest_e = vector_sub_k(r_p_p[tid + 1], r_p_p[tid]);
		float3 e_hat = vector_normalized_k(e);

		float3 force1 = vector_multiply_k(e_hat,(vector_length_k(e)-vector_length_k(rest_e)) * K_S);

		float3 t = multiply_frame_k(s_f[tid - 1], _t[tid]);
		float3 force2 = vector_multiply_k(vector_sub_k(e, t), K_B);
		
		float3 result = vector_add_k(force1, force2);
		p_f[tid] = vector_add_k(p_f[tid], result);
		__syncthreads();
		p_f[tid + 1] = vector_sub_k(p_f[tid + 1], result);
	}
}



//__global__ void integrate_external_hair_force(int *p_i, float3 *p_p, float3 *p_f, float3 *p_v) {
//	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	double dt = 0.0009;
//	float3 gravity = make_float3(0.0, -10, 0.0);
//	for (int i = 0; i < PARTICLE_SIZE; i++) {
//		int tid = p_i[threadIdx.x * PARTICLE_SIZE + i];
//		if (i < PARTICLE_SIZE - 1) {
//			p_f[tid] = vector_add_k(p_f[tid], gravity);
//		}
//		if (i > 0) {
//			float3 ac = p_f[tid];
//			p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
//			p_f[tid] = make_float3(0.0, 0.0, 0.0);
//		}
//	}
//}

__global__ void integrate_external_hair_force(float3 *p_p, float3 *p_f, float3 *p_v) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	double dt = 0.0009;
	float3 gravity = make_float3(0.0, -10, 0.0);


	if (threadIdx.x < blockDim.x - 1) {
		p_f[tid] = vector_add_k(p_f[tid], gravity);
	}

	if (threadIdx.x > 0) {
		float3 ac = p_f[tid];
		p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
		p_f[tid] = make_float3(0.0, 0.0, 0.0);
	}
}

//__global__ void integrate_damping_force(int *p_i, float3 *p_p, float3 *p_f, float3 *p_v) {
//	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	double dt = 0.00009;
//
//	for (int i = 0; i < PARTICLE_SIZE; i++) {
//		int tid = p_i[threadIdx.x * PARTICLE_SIZE + i];
//		if (i < PARTICLE_SIZE - 1) {
//			//Stretch damping
//			float3 d_v = vector_sub_k(p_v[tid + 1], p_v[tid]);
//			float3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
//			float3 e_hat = vector_normalized_k(e);
//
//			float3 force = vector_multiply_k(vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat)), C_S);
//			p_f[tid] = vector_add_k(p_f[tid], force);
//			p_f[tid + 1] = vector_sub_k(p_f[tid + 1], force);
//
//			//Bending damping
//			force = vector_multiply_k(vector_sub_k(d_v, vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat))), C_B);
//			p_f[tid] = vector_add_k(p_f[tid], force);
//			p_f[tid + 1] = vector_sub_k(p_f[tid + 1], force);
//		}
//
//		if (i > 0) {
//			float3 ac = p_f[tid];
//			p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
//			p_f[tid] = make_float3(0.0, 0.0, 0.0);
//		}
//	}
//}

__global__ void integrate_damping_force(float3 *p_p, float3 *p_f, float3 *p_v) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	double dt = 0.00009;
	//if (threadIdx.x > 31) {
	//	__syncthreads();
	//	if (threadIdx.x > 63) {
	//		__syncthreads();
	//		if (threadIdx.x > 95) {
	//			__syncthreads();
	//		}
	//	}
	//}

	if (threadIdx.x < blockDim.x - 1) {
		
		float3 d_v = vector_sub_k(p_v[tid + 1], p_v[tid]);
		float3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
		float3 e_hat = vector_normalized_k(e);

		float3 force1 = vector_multiply_k(vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat)), C_S);
		float3 force2 = vector_multiply_k(vector_sub_k(d_v, vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat))), C_B);

		float3 result = vector_add_k(force1, force2);
		p_f[tid] = vector_add_k(p_f[tid], result);
		__syncthreads();
		p_f[tid + 1] = vector_sub_k(p_f[tid + 1], result);
	}
}

//__global__ void update_position(int *p_i, float3 *p_p, float3 *p_v) {
//	double dt = 0.01;
//	for (int i = 1; i < PARTICLE_SIZE; i++) {
//		int tid = p_i[threadIdx.x * PARTICLE_SIZE + i];
//		p_p[tid] = vector_add_k(p_p[tid], vector_multiply_k(p_v[tid], dt));
//	}
//}


__global__ void update_position(float3 *p_p, float3 *p_v) {
	double dt = 0.01;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIdx.x > 0) {
		p_p[tid] = vector_add_k(p_p[tid], vector_multiply_k(p_v[tid], dt));
	}
}

void HairModel:: simulation() {
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start);

	for (int iter1 = 0; iter1 < 2; iter1++) {
		for (int iter2 = 0; iter2 < 15; iter2++) {
			//cudaMemcpy(p_p_d, s_p_p_d, sizeof(float3) * TOTAL_SIZE, cudaMemcpyDeviceToDevice);
			//smoothing_function_k<< <1, STRAND_SIZE >> > (p_p_d, s_p_p_d, r_p_l_d, 0.23, true);
			//compute_frame_k << <1, STRAND_SIZE >> > (s_f_d, s_p_p_d);

			integrate_internal_hair_force <<<STRAND_SIZE, MAX_SIZE>> > (p_p_d, r_p_p_d, s_f_d, t_d , p_f_d, p_v_d);
			integrate_external_hair_force <<<STRAND_SIZE, MAX_SIZE >> > (p_p_d, p_f_d, p_v_d);
			//integrate << <STRAND_SIZE, PARTICLE_SIZE >> > (p_p_d, p_f_d, p_v_d, 0.0009);

			for (int iter3 = 0; iter3 < 10 * iter2; iter3++) {
				integrate_damping_force <<<STRAND_SIZE, MAX_SIZE >> > (p_p_d, p_f_d, p_v_d);
				integrate << <STRAND_SIZE, MAX_SIZE >> > (p_p_d, p_f_d, p_v_d, 0.00009);
			}
		}
		update_position <<<STRAND_SIZE, MAX_SIZE >> > (p_p_d, p_v_d);
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

