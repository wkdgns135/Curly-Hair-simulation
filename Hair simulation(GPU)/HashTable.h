#ifndef __HASH_TABLE_GPU_H__
#define __HASH_TABLE_GPU_H__

#pragma once
#include <iostream>
#include <cuda_runtime.h>

struct HashParams
{
	float3 world_origin;
	float3 cell_size;

	unsigned int num_cells;
	uint3 grid_size;
};

class HashTable
{
public:
	unsigned int *grid_particle_hash;		// grid hash value for each particle
	unsigned int *grid_particle_index;		// particle index for each particle
	unsigned int *cell_start;				// index of start of each cell in sorted list
	unsigned int *cell_end;				// index of end of cell

public: // for radix sort
	unsigned int* radix_hash;
	unsigned int* radix_index;
public:
	HashTable(void);
	~HashTable(void);
public:
	void		free(void);
	void		init(int number_particles, HashParams *host_params);

};

void		reorderData(HashTable hashing, int numParticles, unsigned int numCells, int numBlock, int numThread);
void		sortParticles(HashTable hashing, int numParticles);
void		updateHashTable(HashTable hashing, float3 *pos, int numParticles, int numBlock, int numThread, HashParams params);

#endif