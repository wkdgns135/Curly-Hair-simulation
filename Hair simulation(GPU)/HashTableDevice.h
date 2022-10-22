#ifndef __HASH_TABLE_H__
#define __HASH_TABLE_H__

#pragma once
#include <iostream>
#include <cuda_runtime.h>

typedef unsigned int uint;

class HashTableDevice
{
public:
	uint *_gridParticleHash;
	uint *_gridParticleIndex;
	uint *_cellStart;
	uint *_cellEnd;

public:
	HashTableDevice(void);
	~HashTableDevice(void);
public:
	void free(void);
	void init(int num_particles, int num_cells);
};

#endif