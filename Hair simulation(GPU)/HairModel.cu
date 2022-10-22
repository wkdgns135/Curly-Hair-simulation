﻿#ifndef __HAIR_MODEL_DEVICE__
#define __HAIR_MODEL_DEVICE__

#include <stdio.h>
#include <cuda_runtime.h>
#include "HairModel.h"
#include "device_launch_parameters.h" 
#include "VectorCalcDevice.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

__constant__ struct Params params;

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

	cudaMalloc((void**)&particle_device.position, sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.velocity, sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.force, sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.r_position, sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.s_position, sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.s_velocity, sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.r_s_position, sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.s_frame, sizeof(Frame) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.t, sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.r_length, sizeof(double) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.R, sizeof(float3) * MAX_SIZE);
	//cudaMalloc((void**)&d,sizeof(float3) * TOTAL_SIZE);

	cudaMemcpy(particle_device.position, particle_host.position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.r_position, particle_host.r_position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.r_s_position, particle_host.r_s_position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.r_s_frame, particle_host.r_s_frame, sizeof(Frame) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.t, particle_host.t, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.r_length, particle_host.r_length, sizeof(double) * STRAND_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.R, particle_host.R, sizeof(float3) * MAX_SIZE, cudaMemcpyHostToDevice);


	array_init << <STRAND_SIZE, MAX_SIZE >> > (particle_device.force);
	array_init << <STRAND_SIZE, MAX_SIZE >> > (particle_device.velocity);
	
	cudaMemcpyToSymbol(params, &params_host, sizeof(Params));
	//array_init << <STRAND_SIZE, MAX_SIZE >> > (d);
}

__global__ void collision_detect(Particle particle, float3 sphere, float radius) {
	if (blockIdx.x == 0)return;
	//if (threadIdx.x == 10)printf("%f %f %f\n", sphere.x, sphere.y, sphere.z);

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float3 &p = particle.position[tid];
	float3 dir = vector_sub_k(p, sphere);
	float dist = vector_length_k(dir);
	if (dist < (radius + 0.01f)) {
		vector_normalize_k(dir);
		float3 new_pos = vector_add_k(sphere, vector_multiply_k(dir, (radius + 0.01f)));
		particle.position[tid] = new_pos;
	}
}

__global__ void integrate(Particle particle, double dt) {
	//if (threadIdx.x > y)return; 
	//if (blockIdx.x > x)return;
	if (threadIdx.x == 0)return;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float3 ac = particle.force[tid];
	particle.velocity[tid] = vector_add_k(particle.velocity[tid], vector_multiply_k(ac, dt));
	particle.force[tid] = make_float3(0.0, 0.0, 0.0);

}

__global__ void integrate_internal_hair_force(Particle particle) {
	if (threadIdx.x > blockDim.x - 2)return;
	//if (blockIdx.x > x)return;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//if(blockIdx.x == 0)printf("thread: %d\n", threadIdx.x);
	float3 e = vector_sub_k(particle.position[tid + 1], particle.position[tid]);
	float3 rest_e = vector_sub_k(particle.r_position[tid + 1], particle.r_position[tid]);
	float3 e_hat = vector_normalized_k(e);
	float3 b = vector_sub_k(particle.s_position[tid + 1], particle.s_position[tid]);
	float3 b_bar = vector_sub_k(particle.r_s_position[tid + 1], particle.r_s_position[tid]);
	float3 b_hat = vector_normalized_k(b);

	//float3 t;
	//if(threadIdx.x == 0) t = multiply_frame_k(s_f[tid], _t[tid]);
	//else t = multiply_frame_k(s_f[tid - 1], _t[tid]);

	float3 force1 = vector_multiply_k(e_hat, (vector_length_k(e) - vector_length_k(rest_e)) * params.K_S);
	float3 force2 = vector_multiply_k(vector_sub_k(e, rest_e), params.K_B);
	float3 force3_1 = vector_multiply_k(b_hat, vector_length_k(b) - vector_length_k(b_bar));
	float3 force3 = vector_multiply_k(force3_1, params.K_C);
	
	float3 force4 = (particle.R[threadIdx.x + 1] - particle.R[threadIdx.x]) * params.R_C;

	float3 result = vector_add_k(force1, force2);
	result = vector_add_k(result, force3);
	result = result + force4;

	particle.force[tid] = vector_add_k(particle.force[tid], result);
	__syncthreads();
	particle.force[tid + 1] = vector_sub_k(particle.force[tid + 1], result);

}

__global__ void integrate_external_hair_force(Particle particle) {
	if (threadIdx.x > blockDim.x - 2)return;
	//if (blockIdx.x > x)return;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	float3 gravity = make_float3(0.0, -10, 0.0);
	particle.force[tid] = vector_add_k(particle.force[tid], gravity);
}

__global__ void integrate_damping_force(Particle particle) {
	if (threadIdx.x > blockDim.x - 2)return;
	//if (blockIdx.x > x)return;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	float3 d_v = vector_sub_k(particle.velocity[tid + 1], particle.velocity[tid]);
	float3 e = vector_sub_k(particle.position[tid + 1], particle.position[tid]);
	float3 e_hat = vector_normalized_k(e);

	float3 b = vector_sub_k(particle.s_position[tid + 1], particle.s_position[tid]);
	float3 b_hat = vector_normalized_k(b);
	float3 v = particle.s_velocity[tid];

	float3 force1 = vector_multiply_k(vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat)), params.C_S);
	float3 force2 = vector_multiply_k(vector_sub_k(d_v, vector_multiply_k(e_hat, vector_dot_k(d_v, e_hat))), params.C_B);
	float3 force3 = vector_multiply_k(b_hat, vector_dot_k(v, b_hat) * params.C_C);


	float3 result = vector_add_k(force1, force2);
	result = vector_add_k(result, force3);

	particle.force[tid] = vector_add_k(particle.force[tid], result);
	__syncthreads();
	particle.force[tid + 1] = vector_sub_k(particle.force[tid + 1], result);

}

__global__ void update_position(Particle particle) {
	//if (threadIdx.x > y)return;
	//if (blockIdx.x > x)return;
	if (threadIdx.x == 0)return;
	double dt = 0.00138883;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	particle.position[tid] = vector_add_k(particle.position[tid], vector_multiply_k(particle.velocity[tid], dt));
}
//float3 *lambda, double *l, double alpha, bool is_position

__global__ void position_smoothing_function_k(float3 *lambda, float3 *dst, float3 *d, double *l, float alpha) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void move_root_up_k(float3 *position) {
	//if (threadIdx.x > y)return;
	//if (blockIdx.x > x)return;
	if (threadIdx.x != 0)return;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	position[tid].y += 0.05;
	double dt = 0.00138883;
}

// added by jhkim
__device__ unsigned int GetGridHash_CurlyHair(int3 gridPos, int3 gridSize)
{
	gridPos.x = gridPos.x & (gridSize.x - 1); 
	gridPos.y = gridPos.y & (gridSize.y - 1);
	gridPos.z = gridPos.z & (gridSize.z - 1);
	return ((gridPos.z * gridSize.y) * gridSize.x) + (gridPos.y * gridSize.x) + gridPos.x;
}

__device__ int3 GetGridPos_CurlyHair(float3 p, float3 cellSize)
{
	int3 gridPos;
	gridPos.x = floor(p.x / 128.0f / cellSize.x);
	gridPos.y = floor(p.y / 128.0f / cellSize.y);
	gridPos.z = floor(p.z / 128.0f / cellSize.z);
	return gridPos;
}

// added by jhkim
__global__ void ReorderData_CurlyHair(HashTableDevice hashing, uint numParticles)
{
	extern __shared__ uint sharedHash[];
	uint index = (blockIdx.x, blockDim.x) + threadIdx.x;
	uint hash;
	if (index < numParticles) {
		hash = hashing._gridParticleHash[index];
		sharedHash[threadIdx.x + 1] = hash;
		if (index > 0 && threadIdx.x == 0) {
			sharedHash[0] = hashing._gridParticleHash[index - 1];
		}
	}
	__syncthreads();
	if (index < numParticles) {
		if (index == 0 || hash != sharedHash[threadIdx.x]) {
			hashing._cellStart[hash] = index;
			if (index > 0) {
				hashing._cellEnd[sharedHash[threadIdx.x]] = index;
			}
		}
		if (index == numParticles - 1) {
			hashing._cellEnd[hash] = index + 1;
		}
	}
}

// added by jhkim
__global__ void UpdateHashKernel_CulryHair(HashTableDevice hashing, Particle particles, int num_particles)
{
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	
	float res = 128.0f;
	float3 dx = make_float3(1.0f / res, 1.0f / res, 1.0f / res);
	int3 gridPos = GetGridPos_CurlyHair(particles.position[index], dx);
	int3 gridSize = make_int3((int)res, (int)res, (int)res);
	unsigned int hash = GetGridHash_CurlyHair(gridPos, gridSize);

	hashing._gridParticleHash[index] = hash;
	hashing._gridParticleIndex[index] = index;
}

//__global__ void ComputeCurlyHairNormalKernel(HashTableDevice hashing, Particle particles)
//{
//	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
//
//	unsigned int p_index = hashing._gridParticleHash[index];
//	float3 p_pos = particles.n_position[p_index];
//	int3 gridPos = GetGridPos_CurlyHair(p_pos, params.cell_size);
//	float3 normal = make_float3(0.0f, 0.0f, 0.0f);
//
//	for (int z = -3; z <= 3; z++) {
//		for (int y = -3; y <= 3; y++) {
//			for (int x = -3; x <= 3; x++) {
//				int3 neighborPos = gridPos + make_int3(x, y, z);
//				uint gridHash = GetGridHash_CurlyHair(neighborPos);
//				uint startIndex = FETCH(hashing._cellStart, gridHash);
//				float force = 0.0f;
//				if (startIndex != 0xffffffff) {
//					uint endIndex = FETCH(hashing._cellEnd, gridHash);
//					for (uint j = startIndex; j < endIndex; j++) {
//						unsigned int np_index = hashing._gridIndex[j];
//						float3 np_pos = particles.n_position[np_index];
//						float np_dens = particles._density[np_index];
//						float3 diff = p_pos - np_pos;
//						//float lenSq = dot(diff, diff);
//						float dx = np_pos.x - p_pos.x;
//						float dy = np_pos.y - p_pos.y;
//						float dz = np_pos.z - p_pos.z;
//						float lenSq = (dx*dx + dy * dy + dz * dz);
//						float w = fmax(1.0 - lenSq / (_CurlyHairParam._particleRadius*_CurlyHairParam._particleRadius), 0.0);
//						normal += (diff * w) / np_dens;
//					}
//				}
//			}
//		}
//	}
//	particles._normal[p_index] = normal;
//}



__global__ void move_root_down_k(float3 *position) {
	//if (threadIdx.x > y)return;
	//if (blockIdx.x > x)return;
	if (threadIdx.x != 0)return;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	position[tid].y -= 0.05;
	double dt = 0.00138883;
}





void HairModel::move_root(int dst) {
	if (dst == 0) {
		move_root_up_k << <STRAND_SIZE, MAX_SIZE >> > (particle_device.position);
	}
	if (dst == 1) {
		move_root_down_k <<<STRAND_SIZE, MAX_SIZE >> > (particle_device.position);
	}
}

void HairModel::updateHashing(void)
{
	int numParticles = TOTAL_SIZE;
	int numBlocks = 70000;
	int numThreads = 1024;
	int gridSize = 128;
	int numCells = gridSize * gridSize * gridSize;

	UpdateHashKernel_CulryHair << <numBlocks, numThreads >> > (_hashing, particle_device, numParticles);
	
	// sort
	//thrust::sort_by_key(thrust::device_ptr<uint>(_hashing._gridParticleHash),
	//	thrust::device_ptr<uint>(_hashing._gridParticleHash + numParticles),
	//	thrust::device_ptr<uint>(_hashing._gridParticleIndex));

	//// reorder
	//cudaMemset(_hashing._cellStart, 0xffffffff, numCells * sizeof(uint));
	//auto sharedMemSize = sizeof(uint)*(numThreads + 1);
	//ReorderData_CurlyHair << <numBlocks, numThreads, sharedMemSize >> > (_hashing, numParticles);
}

void HairModel::freeHashing(void)
{
	_hashing.free();
}

void HairModel::simulation() {
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start);
	
	//updateHashing();

	for (int iter1 = 0; iter1 < 10; iter1++) {
		//collision_detect << <STRAND_SIZE, MAX_SIZE >> > (p_p_d, sphere_pos, sphere_radius, STRAND_SIZE, MAX_SIZE);
		position_smoothing_function(particle_host.position, particle_host.s_position, particle_host.r_length, params_host.A_B, true);
		cudaMemcpy(particle_device.s_position, particle_host.s_position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
		//position_smoothing_function_k << <STRAND_SIZE, MAX_SIZE >> > (p_p_d, s_p_p_d, d,r_p_l_d, A_B);
		//compute_frame(s_f, s_p_p);
		//cudaMemcpy(s_f_d, s_f, sizeof(Frame) * TOTAL_SIZE, cudaMemcpyHostToDevice);
		
		for (int iter2 = 0; iter2 < 15; iter2++) {
			//velocity_smoothing_function_k << <STRAND_SIZE, MAX_SIZE >> > (p_v_d, s_p_v_d, r_p_l_d, A_C);
			cudaMemcpy(particle_host.velocity, particle_device.velocity, sizeof(float3) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
			velocity_smoothing_function(particle_host.velocity, particle_host.s_velocity, particle_host.r_length, params_host.A_C, false);
			cudaMemcpy(particle_device.s_velocity, particle_host.s_velocity, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);

			integrate_internal_hair_force << <STRAND_SIZE, MAX_SIZE >> > (particle_device);
			integrate_external_hair_force << <STRAND_SIZE, MAX_SIZE >> > (particle_device);
			integrate << <STRAND_SIZE, MAX_SIZE >> > (particle_device, 9.25887e-05);
			for (int iter3 = 0; iter3 < 10; iter3++) {
				integrate_damping_force << <STRAND_SIZE, MAX_SIZE >> > (particle_device);
				integrate <<<STRAND_SIZE, MAX_SIZE >>> (particle_device, 9.25887e-06);
			}
		}
		update_position << <STRAND_SIZE, MAX_SIZE >> > (particle_device);
	}

	//freeHashing();


	//performace check

	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float milliseconds = 0.0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//std::cout << " SAXPY execution time : " << milliseconds << " ms " << std::endl;

	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);

	cudaMemcpy(particle_host.position, particle_device.position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
}

#endif // !__HAIR_MODEL_DEVICE__
