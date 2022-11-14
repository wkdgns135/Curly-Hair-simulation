#ifndef __HASH_TABLE_H__
#define __HASH_TABLE_H__

#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "Particle.h"

#define FETCH(t, i) t[i]


class HashTableDevice
{
public:
	unsigned int *_gridHash;	// grid hash value for each particle
	unsigned int *_gridIndex;	// particle index for each particle
	unsigned int *_cellStart;	// index of start of each cell in sorted list
	unsigned int *_cellEnd;		// index of end of cell

public: // for radix sort
	unsigned int* _radixHash;
	unsigned int* _radixIndex;
public:
	HashTableDevice(void);
	~HashTableDevice(void);
public:
	void init(int num_particles, int num_cells);
	void free(void);
};

#endif