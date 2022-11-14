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

void HashTableDevice::init(int number_particles, int number_cells)
{
	cudaMalloc((void **)&_gridHash, number_particles * sizeof(unsigned int));
	cudaMalloc((void **)&_gridIndex, number_particles * sizeof(unsigned int));
	cudaMalloc((void **)&_cellStart, number_cells * sizeof(unsigned int));
	cudaMalloc((void **)&_cellEnd, number_cells * sizeof(unsigned int));

	// radix sort
	//cudaMalloc((void **)&_radixHash, number_particles*sizeof(unsigned int));
	//cudaMalloc((void **)&_radixIndex, number_particles*sizeof(unsigned int));
}

void HashTableDevice::free(void)
{
	cudaFree(_gridHash);
	cudaFree(_gridIndex);
	cudaFree(_cellStart);
	cudaFree(_cellEnd);
	//cudaFree(_radixHash);
	//cudaFree(_radixIndex);
}

