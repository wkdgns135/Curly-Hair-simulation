#include "HashTableDevice.h"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include <cuda_runtime.h>

HashTableDevice::HashTableDevice(void)
{
}

HashTableDevice::~HashTableDevice(void)
{
}

void HashTableDevice::init(int num_particles, int num_cells)
{
	cudaMalloc((void **)&_gridParticleHash, num_particles * sizeof(uint));
	cudaMalloc((void **)&_gridParticleIndex, num_particles * sizeof(uint));
	cudaMalloc((void**)&_cellStart, num_cells * sizeof(uint));
	cudaMalloc((void**)&_cellEnd, num_cells * sizeof(uint));
	printf("init hashing\n");
}

void HashTableDevice::free(void)
{
	cudaFree(_gridParticleHash);
	cudaFree(_gridParticleIndex);
	cudaFree(_cellStart);
	cudaFree(_cellEnd);
}