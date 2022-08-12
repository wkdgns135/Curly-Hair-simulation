#pragma once
#include <stdio.h>
#include "HairModel.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h" 
#include "vector_calc.cuh"


void HairModel::device_init() {
	cudaMalloc((void**)&p_i_d,sizeof(int) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&p_p_d,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&p_v_d,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&p_f_d,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&r_p_p_d,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&s_p_p_d,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	
	//cudaMalloc((void**)&r_s_p_p,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&s_f_d,sizeof(Frame) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&t_d,sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE);
	cudaMalloc((void**)&r_p_l_d,sizeof(double) * STRAND_SIZE * PARTICLE_SIZE);
	
	cudaMemcpy(p_i_d, p_i, sizeof(int) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(p_p_d, p_p, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_p_p_d, r_p_p, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(r_s_f, h.r_s_f, sizeof(Frame) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	//cudaMemcpy(r_s_p_p, h.r_s_p_p, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(t_d, t, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(r_p_l_d, r_p_l, sizeof(double) * STRAND_SIZE, cudaMemcpyHostToDevice);


	array_init << <STRAND_SIZE, PARTICLE_SIZE >> > (p_f_d);
	array_init << <STRAND_SIZE, PARTICLE_SIZE >> > (p_v_d);
}

//__global__ void integrate_internal_hair_force(int *p_i,double3 *p_p, double3 *r_p_p, Frame *s_f, double3* _t,double3 *p_f, double3 *p_v) {
//	double dt = 0.0009;
//	
//	for (int i = 0; i < PARTICLE_SIZE; i++) {
//		int tid = p_i[threadIdx.x * PARTICLE_SIZE + i];
//		if (i < PARTICLE_SIZE - 1) {
//			//Stretch spring
//			double3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
//			double3 rest_e = vector_sub_k(r_p_p[tid + 1], r_p_p[tid]);
//			double3 e_hat = vector_normalized_k(e);
//
//			double3 force = vector_multiply_k(e_hat, (vector_length_k(e) - vector_length_k(rest_e)) * K_S);
//
//			p_f[tid] = vector_add_k(p_f[tid], force);
//			p_f[tid + 1] = vector_sub_k(p_f[tid + 1], force);
//
//			//Bending spring
//			double3 t = multiply_frame_k(s_f[tid - 1], _t[tid]);
//			force = vector_multiply_k(vector_sub_k(e, t), K_B);
//			
//			p_f[tid] = vector_add_k(p_f[tid], force);
//			p_f[tid + 1] = vector_sub_k(p_f[tid + 1], force);
//			
//
//		}
//		if (i > 0) {
//			double3 ac = p_f[tid];
//			p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
//			p_f[tid] = make_double3(0.0, 0.0, 0.0);
//		}
//	}
//}

__global__ void integrate(double3 *p_p, double3 *p_f, double3 *p_v, double dt) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x > 0) {
		double3 ac = p_f[tid];
		p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
		p_f[tid] = make_double3(0.0, 0.0, 0.0);
	}
}

__global__ void integrate_internal_hair_force(double3 *p_p, double3 *r_p_p, Frame *s_f, double3* _t, double3 *p_f, double3 *p_v) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	double dt = 0.0009;

	//if(blockIdx.x == 0)printf("thread: %d\n", threadIdx.x);
	double3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
	double3 rest_e = vector_sub_k(r_p_p[tid + 1], r_p_p[tid]);
	double3 e_hat = vector_normalized_k(e);

	double3 force1 = vector_multiply_k(e_hat,(vector_length_k(e)-vector_length_k(rest_e)) * K_S);

	double3 t = multiply_frame_k(s_f[tid - 1], _t[tid]);
	double3 force2 = vector_multiply_k(vector_sub_k(e, t), K_B);
		
	double3 result = vector_add_k(force1, force2);
	p_f[tid] = vector_add_k(p_f[tid], result);
	__syncthreads();
	p_f[tid + 1] = vector_sub_k(p_f[tid + 1], result);
	
}



//__global__ void integrate_external_hair_force(int *p_i, double3 *p_p, double3 *p_f, double3 *p_v) {
//	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	double dt = 0.0009;
//	double3 gravity = make_double3(0.0, -10, 0.0);
//	for (int i = 0; i < PARTICLE_SIZE; i++) {
//		int tid = p_i[threadIdx.x * PARTICLE_SIZE + i];
//		if (i < PARTICLE_SIZE - 1) {
//			p_f[tid] = vector_add_k(p_f[tid], gravity);
//		}
//		if (i > 0) {
//			double3 ac = p_f[tid];
//			p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
//			p_f[tid] = make_double3(0.0, 0.0, 0.0);
//		}
//	}
//}

__global__ void integrate_external_hair_force(double3 *p_p, double3 *p_f, double3 *p_v) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	double dt = 0.0009;
	double3 gravity = make_double3(0.0, -10, 0.0);
	if (threadIdx.x < PARTICLE_SIZE - 1) {
		p_f[tid] = vector_add_k(p_f[tid], gravity);
	}

	//if (threadIdx.x > 0) {
	//	double3 ac = p_f[tid];
	//	p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
	//	p_f[tid] = make_double3(0.0, 0.0, 0.0);
	//}
}

//__global__ void integrate_damping_force(int *p_i, double3 *p_p, double3 *p_f, double3 *p_v) {
//	//int tid = blockIdx.x * blockDim.x + threadIdx.x;
//	double dt = 0.00009;
//
//	for (int i = 0; i < PARTICLE_SIZE; i++) {
//		int tid = p_i[threadIdx.x * PARTICLE_SIZE + i];
//		if (i < PARTICLE_SIZE - 1) {
//			//Stretch damping
//			double3 d_v = vector_sub_k(p_v[tid + 1], p_v[tid]);
//			double3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
//			double3 e_hat = vector_normalized_k(e);
//
//			double3 force = vector_multiply_k(vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat)), C_S);
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
//			double3 ac = p_f[tid];
//			p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
//			p_f[tid] = make_double3(0.0, 0.0, 0.0);
//		}
//	}
//}

__global__ void integrate_damping_force(double3 *p_p, double3 *p_f, double3 *p_v) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	double dt = 0.00009;

	if (threadIdx.x < PARTICLE_SIZE - 1) {
		
		double3 d_v = vector_sub_k(p_v[tid + 1], p_v[tid]);
		double3 e = vector_sub_k(p_p[tid + 1], p_p[tid]);
		double3 e_hat = vector_normalized_k(e);

		double3 force1 = vector_multiply_k(vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat)), C_S);
		double3 force2 = vector_multiply_k(vector_sub_k(d_v, vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat))), C_B);

		double3 result = vector_add_k(force1, force2);
		p_f[tid] = vector_add_k(p_f[tid], result);
		__syncthreads();
		p_f[tid + 1] = vector_sub_k(p_f[tid + 1], result);

	}

	//if (threadIdx.x > 0) {
	//	double3 ac = p_f[tid];
	//	p_v[tid] = vector_add_k(p_v[tid], vector_multiply_k(ac, dt));
	//	p_f[tid] = make_double3(0.0, 0.0, 0.0);
	//}
}

//__global__ void update_position(int *p_i, double3 *p_p, double3 *p_v) {
//	double dt = 0.01;
//	for (int i = 1; i < PARTICLE_SIZE; i++) {
//		int tid = p_i[threadIdx.x * PARTICLE_SIZE + i];
//		p_p[tid] = vector_add_k(p_p[tid], vector_multiply_k(p_v[tid], dt));
//	}
//}


__global__ void update_position(double3 *p_p, double3 *p_v) {
	double dt = 0.01;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIdx.x > 0) {
		p_p[tid] = vector_add_k(p_p[tid], vector_multiply_k(p_v[tid], dt));
	}
}

void HairModel:: simulation() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	for (int iter1 = 0; iter1 < 2; iter1++) {
		for (int iter2 = 0; iter2 < 15; iter2++) {
			//cudaMemcpy(p_p_d, s_p_p_d, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyDeviceToDevice);
			//smoothing_function_k<< <1, STRAND_SIZE >> > (p_p_d, s_p_p_d, r_p_l_d, 0.23, true);
			//compute_frame_k << <1, STRAND_SIZE >> > (s_f_d, s_p_p_d);

			integrate_internal_hair_force <<<STRAND_SIZE, PARTICLE_SIZE>> > (p_p_d, r_p_p_d, s_f_d, t_d , p_f_d, p_v_d);
			integrate << <STRAND_SIZE, PARTICLE_SIZE >> > (p_p_d, p_f_d, p_v_d, 0.0009);

			integrate_external_hair_force <<<STRAND_SIZE, PARTICLE_SIZE >> > (p_p_d, p_f_d, p_v_d);
			integrate << <STRAND_SIZE, PARTICLE_SIZE >> > (p_p_d, p_f_d, p_v_d, 0.0009);

			for (int iter3 = 0; iter3 < 10 * iter2; iter3++) {
				integrate_damping_force <<<STRAND_SIZE, PARTICLE_SIZE>> > (p_p_d, p_f_d, p_v_d);
				integrate << <STRAND_SIZE, PARTICLE_SIZE >> > (p_p_d, p_f_d, p_v_d, 0.00009);
			}
		}
		update_position <<<STRAND_SIZE, PARTICLE_SIZE >> > (p_p_d, p_v_d);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << " SAXPY execution time : " << milliseconds << " ms " << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaMemcpy(p_p, p_p_d, sizeof(double3) * STRAND_SIZE * PARTICLE_SIZE, cudaMemcpyDeviceToHost);
}

