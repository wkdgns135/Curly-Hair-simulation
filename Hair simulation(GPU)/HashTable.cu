#include "HashTable.h"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include <cuda_runtime.h>
#include "helper_math.h"
#include <helper_cuda.h>

__constant__ HashParams params;

HashTable::HashTable(void)
{
	
}

HashTable::~HashTable(void)
{
	free();
}

void HashTable::init(int number_particles, HashParams *host_params)
{

	cudaMalloc((void **)&grid_particle_hash, number_particles*sizeof(unsigned int));
	cudaMalloc((void **)&grid_particle_index, number_particles*sizeof(unsigned int));
	cudaMalloc((void **)&cell_start, host_params->num_cells*sizeof(unsigned int));
	cudaMalloc((void **)&cell_end, host_params->num_cells*sizeof(unsigned int));
	
	// radix sort
	cudaMalloc((void **)&radix_hash, number_particles*sizeof(unsigned int));
	cudaMalloc((void **)&radix_index, number_particles*sizeof(unsigned int));

	checkCudaErrors(cudaMemcpyToSymbol(params, host_params, sizeof(HashTable)));
}

void HashTable::free(void)
{
	cudaFree(grid_particle_hash);
	cudaFree(grid_particle_index);
	cudaFree(cell_start);
	cudaFree(cell_end);
	cudaFree(radix_hash);
	cudaFree(radix_index);
}


__device__ int3 GetGridPos(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - params.world_origin.x) / params.cell_size.x);
	gridPos.y = floor((p.y - params.world_origin.y) / params.cell_size.y);
	gridPos.z = floor((p.z - params.world_origin.z) / params.cell_size.z);
	return gridPos;
}

__device__ unsigned int GetGridHash(int3 gridPos)
{
	gridPos.x = gridPos.x & (params.grid_size.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.grid_size.y - 1);
	gridPos.z = gridPos.z & (params.grid_size.z - 1);

	return __umul24(__umul24(gridPos.z, params.grid_size.y), params.grid_size.x) + __umul24(gridPos.y, params.grid_size.x) + gridPos.x;
}

void sortParticles(HashTable hashing, int numParticles)
{
	thrust::sort_by_key(thrust::device_ptr<uint>((uint*)(hashing.grid_particle_hash)),
		thrust::device_ptr<uint>((uint*)(hashing.grid_particle_hash + numParticles)),
		thrust::device_ptr<uint>((uint*)(hashing.grid_particle_index)));
}

void reorderData(HashTable hashing, int numParticles, unsigned int numCells, int numBlock, int numThread)
{
	cudaMemset(hashing.cell_start, 0xffffffff, numCells * sizeof(unsigned int));
	unsigned int smemSize = sizeof(unsigned int)*(numThread + 1);
	ReorderDataKernel << < numBlock, numThread, smemSize >> > (hashing, numParticles);
}

void updateHashTable(HashTable hashing, float3 *pos, int numParticles, int numBlock, int numThread, HashParams params)
{
	UpdateHashTableKernel << < numBlock, numThread >> > (hashing, particles, numParticles);
	sortParticles(hashing, numParticles);
	reorderData(hashing, numParticles, params.num_cells, numBlock, numThread);
}



__global__ void ReorderDataKernel(HashTable hashing, uint numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = hashing.grid_particle_hash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = hashing.grid_particle_hash[index - 1];
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
			hashing.cell_start[hash] = index;

			if (index > 0)
				hashing.cell_end[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			hashing.cell_end[hash] = index + 1;
		}
	}
}


__global__ void UpdateHashTableKernel(HashTable hashing, float3 *pos, int numParticles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;

	// get address in grid
	int3 gridPos = GetGridPos(pos[index]);
	uint hash = GetGridHash(gridPos);

	// store grid hash and particle index
	hashing.grid_particle_hash[index] = hash;
	hashing.grid_particle_index[index] = index;
}