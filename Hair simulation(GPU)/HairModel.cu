//#pragma once
//#include "HairModel.cuh"
//#include "vector_calc.h"
//
//
//void cuda_init(HairModel h) {
//	for (int i = 0; i < STRAND_SIZE; i++) {
//		cudaMalloc(&(h.s_d[STRAND_SIZE].r_p_p), sizeof(double) * 3 * PARTICLE_SIZE);
//		cudaMemcpy(h.s[STRAND_SIZE].r_p_p, h.s_d[STRAND_SIZE].r_p_p, sizeof(double) * 3 * PARTICLE_SIZE, cudaMemcpyDeviceToHost);
//
//		cudaMalloc(&(h.s_d[STRAND_SIZE].r_p_p), sizeof(double) * 3 * PARTICLE_SIZE);
//		cudaMemcpy(h.s[STRAND_SIZE].r_p_p, h.s_d[STRAND_SIZE].r_p_p, sizeof(double) * 3 * PARTICLE_SIZE, cudaMemcpyDeviceToHost);
//		
//		cudaMalloc(&(h.s_d[STRAND_SIZE].r_p_p), sizeof(double) * 3 * PARTICLE_SIZE);
//		cudaMemcpy(h.s[STRAND_SIZE].r_p_p, h.s_d[STRAND_SIZE].r_p_p, sizeof(double) * 3 * PARTICLE_SIZE, cudaMemcpyDeviceToHost);
//	}
//}
//
//void simulation() {
//
//}