#ifndef __HAIR_MODEL_DEVICE__
#define __HAIR_MODEL_DEVICE__

#include <stdio.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include "HairModel.h"
#include "device_launch_parameters.h" 
#include "VectorCalcDevice.cuh"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

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
	cudaMalloc((void**)&particle_device.n_position, sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.density, sizeof(float) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.saturation, sizeof(float) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.test_pos, sizeof(float3) * TOTAL_SIZE);
	cudaMalloc((void**)&particle_device.wet_position, sizeof(float3) * TOTAL_SIZE);
	//cudaMalloc((void**)&d,sizeof(float3) * TOTAL_SIZE);

	cudaMemcpy(particle_device.position, particle_host.position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.r_position, particle_host.r_position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.r_s_position, particle_host.r_s_position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.r_s_frame, particle_host.r_s_frame, sizeof(Frame) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.t, particle_host.t, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.r_length, particle_host.r_length, sizeof(double) * STRAND_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.R, particle_host.R, sizeof(float3) * MAX_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_device.saturation, particle_host.saturation, sizeof(float) * TOTAL_SIZE, cudaMemcpyHostToDevice);

	cudaMemcpy(particle_device.n_position, particle_host.n_position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(particle_host.wet_position, particle_host.n_position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToHost);


	array_init << <STRAND_SIZE, MAX_SIZE >> > (particle_device.force);
	array_init << <STRAND_SIZE, MAX_SIZE >> > (particle_device.velocity);

	cudaMemcpyToSymbol(params, &params_host, sizeof(Params));
}

void HairModel::device_free() {
	cudaFree(particle_device.position);
	cudaFree(particle_device.velocity);
	cudaFree(particle_device.force);
	cudaFree(particle_device.r_position);
	cudaFree(particle_device.s_position);
	cudaFree(particle_device.s_velocity);
	cudaFree(particle_device.r_s_position);
	cudaFree(particle_device.s_frame);
	cudaFree(particle_device.t);
	cudaFree(particle_device.r_length);
	cudaFree(particle_device.R);
	cudaFree(particle_device.n_position);
	cudaFree(particle_device.test_pos);
	cudaFree(particle_device.density);
	cudaFree(particle_device.saturation);
	cudaFree(particle_device.wet_position);
	cudaFree(&params);
}

void HairModel::set_parameter() {
	cudaMemcpyToSymbol(params, &params_host, sizeof(Params));
}

__global__ void integrate(Particle particle, double dt) {
	//if (threadIdx.x > y)return; 
	//if (blockIdx.x > x)return;
	if (threadIdx.x == 0)return;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float3 ac = particle.force[tid];

	particle.velocity[tid] = vector_add_k(particle.velocity[tid], vector_multiply_k(ac, dt));
	particle.force[tid] = make_float3(0.0, 0.0, 0.0);

	//apply stiffness  
	float staticStiff = 0.999995f;
	particle.velocity[tid] = particle.velocity[tid] * staticStiff;

	// apply dynamic-stiffness for wetting
	float stiffMin = 0.9999f;
	float stiffMax = 1.0f;
	float dynamicStiff = (particle.saturation[tid]) * stiffMin + (1.0 - particle.saturation[tid]) * stiffMax;
	particle.velocity[tid] = particle.velocity[tid] * dynamicStiff;
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

	//internal_hair_force 
	float3 force1 = vector_multiply_k(e_hat, (vector_length_k(e) - vector_length_k(rest_e)) * params.K_S);
	float3 force2 = vector_multiply_k(vector_sub_k(e, rest_e), params.K_B);
	float3 force3_1 = vector_multiply_k(b_hat, vector_length_k(b) - vector_length_k(b_bar));
	float3 force3 = vector_multiply_k(force3_1, params.K_C);
	
	//Curl force
	float3 particle_root_tip = particle.position[(blockIdx.x + 1) * blockDim.x - 1] - particle.position[blockIdx.x * blockDim.x];
	particle_root_tip = vector_normalized_k(particle_root_tip);
	float3 helix_root_tip = make_float3(0.0, -1.0, 0.0);
	matrix3 rotmat = rot_mat_from_two_vectors(particle_root_tip, helix_root_tip);

	float r_c = params.R_C * particle.saturation[tid] * particle.saturation[tid];
	//force2 = vector_multiply_k(force2 , (1 - particle.saturation[tid])); //bending spring
	float3 ref_vec = (particle.R[threadIdx.x + 1] - particle.R[threadIdx.x]);
	ref_vec = rot_vec_by_mat(ref_vec, rotmat);

	float3 force4 = ref_vec *  r_c;
	float3 result = force1 + force2 + force3 + force4;


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
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIdx.x > blockDim.x - 2) return;
	//if (blockIdx.x > x)return;
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


__global__ void move_root_down_k(float3 *position) {
	//if (threadIdx.x > y)return;
	//if (blockIdx.x > x)return;
	if (threadIdx.x != 0)return;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	position[tid].y -= 0.05;
	double dt = 0.00138883;
}

__device__ int3 GetGridPosCurlyHair(float3 p)
{
	int3 gridPos;
	gridPos.x = floor(p.x / params.cell_size.x);
	gridPos.y = floor(p.y / params.cell_size.y);
	gridPos.z = floor(p.z / params.cell_size.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ unsigned int GetGridHashCurlyHair(int3 gridPos)
{
	gridPos.x = gridPos.x & (params.grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.grid_size.y - 1);
	gridPos.z = gridPos.z & (params.grid_size.z - 1);
	return __umul24(__umul24(gridPos.z, params.grid_size.y), params.grid_size.x) + __umul24(gridPos.y, params.grid_size.x) + gridPos.x;
}

__global__ void ReorderCurlyHairDataKernel(HashTableDevice hashing, int numParticles)
{
	extern __shared__ unsigned int sharedHash[];    // blockSize + 1 elements
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	unsigned int hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = hashing._gridHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = hashing._gridHash[index - 1];
		}
	}

	__syncthreads();

	if (index < numParticles)
	{
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			hashing._cellStart[hash] = index;

			if (index > 0)
				hashing._cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			hashing._cellEnd[hash] = index + 1;
		}
	}
}

__global__ void UpdateHashKernel_CulryHair(HashTableDevice hashing, Particle particles, int num_particles)
{
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;

	// get address in grid
	int3 gridPos = GetGridPosCurlyHair(particles.n_position[index]);
	unsigned int hash = GetGridHashCurlyHair(gridPos);

	// store grid hash and particle index
	hashing._gridHash[index] = hash;
	hashing._gridIndex[index] = index;
}

__global__ void ComputeCurlyHairDensityKernel(HashTableDevice hashing, Particle particles)
{
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	unsigned int p_index = hashing._gridIndex[index];
	float3 p_normPos = particles.n_position[p_index];
	int3 gridPos = GetGridPosCurlyHair(p_normPos);
	float density = 0.0f;

	for (int z = -1; z <= 1; z++) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				int3 neighborPos = gridPos + make_int3(x, y, z);
				unsigned int gridHash = GetGridHashCurlyHair(neighborPos);
				unsigned int startIndex = FETCH(hashing._cellStart, gridHash);
				float force = 0.0f;
				if (startIndex != 0xffffffff) {
					unsigned int endIndex = FETCH(hashing._cellEnd, gridHash);
					for (unsigned int j = startIndex; j < endIndex; j++) {
						unsigned int np_index = hashing._gridIndex[j];
						float3 np_normPos = particles.n_position[np_index];
						float3 diff = np_normPos - p_normPos;
						float lenSq = vector_dot_k(diff, diff);
						float w = fmax(1.0 - lenSq / (params.particle_radius*params.particle_radius), 0.0);
						force += w;
					}
				}
				density += force;
			}
		}
	}
	particles.density[p_index] = density;
}


__global__ void DeformWetCurlyHairKernel(HashTableDevice hashing, Particle particles)
{
	//if (threadIdx.x > 1) return;
	unsigned int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	unsigned int p_index = hashing._gridIndex[index];
	float3 p_pos = particles.n_position[p_index];
	int3 gridPos = GetGridPosCurlyHair(p_pos);
	float p_sat = particles.saturation[p_index];
	particles.wet_position[p_index] = p_pos;

	if (p_sat <= 0.0f) return;

	int r = 6;
	float3 sat_pos = make_float3(0, 0, 0);
	int size = 0;

	for (int z = -r; z <= r; z++) {
		for (int y = -r; y <= r; y++) {
			for (int x = -r; x <= r; x++) {
				int3 neighborPos = gridPos + make_int3(x, y, z);
				unsigned int gridHash = GetGridHashCurlyHair(neighborPos);
				unsigned int startIndex = FETCH(hashing._cellStart, gridHash);
				if (startIndex != 0xffffffff) {
					unsigned int endIndex = FETCH(hashing._cellEnd, gridHash);
					for (unsigned int j = startIndex; j < endIndex; j++) {
						unsigned int np_index = hashing._gridIndex[j];
						float3 np_pos = particles.n_position[np_index];
						float3 diff = p_pos - np_pos;
						float dist = vector_length_k(diff);
						float np_sat = particles.saturation[np_index];
						float range = params.particle_radius * (r * 0.5);
						if (np_sat > 0.0f && dist < range) {
							float inverse_sat = (1.0f - np_sat);
							float3 approx_pos = np_pos * np_sat + p_pos * inverse_sat;
							sat_pos = sat_pos +approx_pos;
							size++;
						}
					}
				}
			}
		}
	}

	if (size == 0) return;

	float fsize = (float)size;
	sat_pos = sat_pos / fsize;
	float tightness = 0.85f;
	float3 displacement = (sat_pos - p_pos) * tightness;
	particles.wet_position[p_index] = particles.wet_position[p_index] + displacement;
}

__global__ void bouncing_test_k(Particle particle, float offset) {
	if (threadIdx.x != 0)return;
	int tid = blockIdx.x * blockDim.x;
	particle.position[tid].y += sin(offset * 0.25) * 0.5;

};

__global__ void rotating_test_k(Particle particle, matrix3 rot) {
	if (threadIdx.x != 0) return;
	int tid = blockIdx.x * blockDim.x;
	float3 pos = rot_vec_by_mat(particle.position[tid], rot);
	particle.position[tid] = pos;
}

__global__ void collision_test_k(Particle particle, float dt)
{
	if (threadIdx.x == 0) return;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float3 pos = particle.n_position[tid];
	float3 vel = particle.velocity[tid];
	float3 normal = pos - params.sphere_pos;
	normal = vector_normalized_k(normal);
	float sdf = powf(pos.x - params.sphere_pos.x, 2.0f) + powf(pos.y - params.sphere_pos.y, 2.0f) + powf(pos.z - params.sphere_pos.z, 2.0f) - (params.sphere_rad * params.sphere_rad);
	dt = dt * 0.1;
	float phi = sdf + dt * vector_dot_k(vel, normal);

	if (phi < 0.0) {
		float vn = vector_dot_k(vel, normal);
		float3 vt = vel - normal * vn;
		float v_new = vn - phi / dt;
		float mu = 0.3f;
		float friction = 1.0f - mu * ((v_new - vn) / vector_length_k(vt));
		float3 v_rel;
		if (friction < 0.0) {
			v_rel = make_float3(0.0, 0.0, 0.0);
		}
		else {
			v_rel = vt * friction;
		}
		particle.velocity[tid] = normal * v_new + v_rel;
	}
}

__device__ float3 AdhesionCurlyHair(float r, float h, float3 dir)
{
	float half_h = h / 2.0f;
	float3 force = make_float3(0.0f, 0.0f , 0.0f);
	if (r <= half_h) {
		float w = 0.007f / powf(h, 3.25f);
		float value = -(4.0f * r * r) / h + 6.0f * r - 2.0f * h;
		force = dir * w * value / r;
		return force;
	}
	return force;
}

__global__ void adhesion_test_k(Particle particle, float dt)
{
	if (threadIdx.x == 0) return;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float3 pos = particle.n_position[tid];
	float3 vel = particle.velocity[tid];
	float3 normal = pos - params.sphere_pos;
	normal = vector_normalized_k(normal);
	float sdf = powf(pos.x - params.sphere_pos.x, 2.0f) + powf(pos.y - params.sphere_pos.y, 2.0f) + powf(pos.z - params.sphere_pos.z, 2.0f) - ((params.sphere_rad + 0.01) * (params.sphere_rad + 0.01));
	dt = dt * 0.1;
	float phi = sdf + dt * vector_dot_k(vel, normal);

	if (phi < 0.0) {
		float vn = vector_dot_k(vel, normal);
		float3 vt = vel - normal * vn;
		float v_new = vn - phi / dt;
		float mu = 0.3f;
		float friction = 1.0f - mu * ((v_new - vn) / vector_length_k(vt));
		float3 v_rel;
		if (friction < 0.0) {
			v_rel = make_float3(0.0, 0.0, 0.0);
		}
		else {
			v_rel = vt * friction;
		}
		particle.velocity[tid] = normal * v_new + v_rel;

		// apply adhesion force
		float dist = fabs(phi); // Penetration depth
		float weight = 1300.0f;
		float3 diff = normal * dist;
		float3 adhesion = AdhesionCurlyHair(dist, params.particle_radius, diff);
		float mass = 1.0f;
		float3 adhesionForce = adhesion * weight * mass;
		particle.velocity[tid] = particle.velocity[tid] + adhesionForce * dt;
	}
}



void HairModel::bouncing_test() {
	bouncing_test_k << <STRAND_SIZE, MAX_SIZE >> > (particle_device, bouncing_offset);
	bouncing_offset += 1;
}

void HairModel::rotating_test(void)
{
	auto radian = cos(rotating_offset) * 1.0;
	auto theta = radian * 0.017453292519943295769236907684886;

	float3 minB = min_b;
	float3 maxB = max_b;
	//float3 pivot =  vector_multiply_k((maxB + minB) , 0.5);
	float3 normal = make_float3(0.0, 1.0, 0.0);
	matrix3 rotM;
	set_identity(rotM);

	// compute rotate matrix
	rotM.x.x = cos(theta) + (normal.x*normal.x)*(1.0 - cos(theta));
	rotM.x.y = (normal.x*normal.y)*(1.0 - cos(theta)) - normal.z*sin(theta);
	rotM.x.z = (normal.x*normal.z)*(1.0 - cos(theta)) + normal.y*sin(theta);
	rotM.y.x = (normal.y*normal.x)*(1.0 - cos(theta)) + normal.z*sin(theta);
	rotM.y.y = cos(theta) + (normal.y*normal.y)*(1.0 - cos(theta));
	rotM.y.z = (normal.y*normal.z)*(1.0 - cos(theta)) - normal.x*sin(theta);
	rotM.z.x = (normal.z*normal.x)*(1.0 - cos(theta)) - normal.y*sin(theta);
	rotM.z.y = (normal.z*normal.y)*(1.0 - cos(theta)) + normal.x*sin(theta);
	rotM.z.z = cos(theta) + (normal.z*normal.z)*(1.0 - cos(theta));

	rotating_test_k<<<STRAND_SIZE, MAX_SIZE>>> (particle_device, rotM);
	rotating_offset += 0.02; // angle range
}

void HairModel::collision_test() {
	collision_test_k << <STRAND_SIZE, MAX_SIZE >> > (particle_device, 0.001f);
}

void HairModel::adhesion_test() {
	adhesion_test_k << <STRAND_SIZE, MAX_SIZE >> > (particle_device, 0.001f);
}

void HairModel::sphere_moving() {
	auto w = cos(sphere_offset) * 0.018f;
	params_host.sphere_pos.y += w;
	sphere_offset += 0.05f;
	set_parameter();
}

void HairModel::move_root(int dst) {
	if (dst == 0) {
		move_root_up_k << <STRAND_SIZE, MAX_SIZE >> > (particle_device.position);
	}
	if (dst == 1) {
		move_root_down_k <<<STRAND_SIZE, MAX_SIZE >> > (particle_device.position);
	}
}

void HairModel::computeMaxDensity(void)
{
	updateHashing();
	updateDensity();
	params_host.max_density = -std::numeric_limits<float>::infinity();

	for (int i = 0; i < TOTAL_SIZE; i++) {
		params_host.max_density = fmax(params_host.max_density, particle_host.density[i]);
	}
	set_parameter();

	_hashing.free();
	printf("maxium density : %f\n", params_host.max_density);
}

void HairModel::updateHashing(void)
{
	_hashing.init(TOTAL_SIZE, params_host.num_cells);
	UpdateHashKernel_CulryHair << <STRAND_SIZE, MAX_SIZE >> > (_hashing, particle_device, TOTAL_SIZE);

	thrust::sort_by_key(thrust::device_ptr<unsigned int>((unsigned int*)(_hashing._gridHash)),
		thrust::device_ptr<unsigned int>((unsigned int*)(_hashing._gridHash + TOTAL_SIZE)),
		thrust::device_ptr<unsigned int>((unsigned int*)(_hashing._gridIndex)));

	cudaMemset(_hashing._cellStart, 0xffffffff, params_host.num_cells * sizeof(unsigned int));
	int smemSize = sizeof(unsigned int)*(MAX_SIZE + 1);
	ReorderCurlyHairDataKernel << < STRAND_SIZE, MAX_SIZE, smemSize >> > (_hashing, TOTAL_SIZE);
}

void HairModel::updateDensity(void) {
	ComputeCurlyHairDensityKernel << <STRAND_SIZE, MAX_SIZE >> > (_hashing, particle_device);
	cudaMemcpy(particle_host.density, particle_device.density, sizeof(float) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
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

	if (state == COHESION_TEST) {
		updateHashing();
		updateDensity();
	}

	if (state == BOUNCING_TEST)bouncing_test();
	if (state == ROTATE_TEST)rotating_test();
	if (state == COLLISION_TEST || state == ADHESION_TEST)sphere_moving();
	
	for (int iter1 = 0; iter1 < 10; iter1++) {

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

		if (state == COLLISION_TEST)collision_test();
		if (state == ADHESION_TEST)adhesion_test();
		update_position << <STRAND_SIZE, MAX_SIZE >> > (particle_device);
		
		if (state == COHESION_TEST) {
			DeformWetCurlyHairKernel << <STRAND_SIZE, MAX_SIZE >> > (_hashing, particle_device);
			cudaMemcpy(particle_host.wet_position, particle_device.wet_position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
		}
		cudaMemcpy(particle_host.position, particle_device.position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
	}

	//cudaMemcpy(particle_host.test_pos, particle_device.test_pos, sizeof(float3) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
	normalize_position();
	cudaMemcpy(particle_device.n_position, particle_host.n_position, sizeof(float3) * TOTAL_SIZE, cudaMemcpyHostToDevice);
	if (state == COHESION_TEST)freeHashing();
}

#endif // !__HAIR_MODEL_DEVICE__
