#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "utils.h"

// Global variables
int number_multi_processors;
int number_blocks;
int number_threads;
int max_threads_per_mp;
int batch_size;
int device_count; 

int gcd(int a, int b)
{
    return (a == 0) ? b : gcd(b % a, a);
}

extern "C" void gpu_init(uint32_t batchsize)
{
    cudaError_t cudaerr = cudaGetDeviceCount(&device_count);
    if (cudaerr != cudaSuccess)
    {
        printf("cudaGetDeviceCount failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    printf("Found %d CUDA devices\n", device_count);

    number_multi_processors = 0;
    max_threads_per_mp = 0;

    for (int device = 0; device < device_count; ++device)
    {
        cudaDeviceProp device_prop;
        cudaerr = cudaGetDeviceProperties(&device_prop, device);
        if (cudaerr != cudaSuccess)
        {
            printf("Getting properties for device %d failed with error \"%s\".\n", device, cudaGetErrorString(cudaerr));
            exit(EXIT_FAILURE);
        }

        printf("Device %d: \"%s\"\n", device, device_prop.name);

        number_multi_processors += device_prop.multiProcessorCount;
        max_threads_per_mp += device_prop.maxThreadsPerMultiProcessor;

        if (device == 0)
        {
            number_threads = std::min(device_prop.maxThreadsPerBlock, 256);
        }

        cudaSetDevice(device);
    }

    int block_size = max_threads_per_mp / gcd(max_threads_per_mp, number_threads);
    number_blocks = block_size * number_multi_processors;
    batch_size = batchsize;
}

__device__ uint64_t saturating_add(uint64_t a, uint64_t b)
{
    uint64_t result = a + b;
    if (result < a)
    {
        return UINT64_MAX;
    }
    return result;
}

__device__ uint32_t difficulty(const uint8_t *hash)
{
    uint32_t count = 0;
    for (int i = 0; i < 32; i++)
    {
        uint32_t lz = __clz((int)hash[i]) - 24; // __clz counts leading zeros of a 32-bit int, adjust for 8-bit value

        count += lz;
        if (lz < 8)
        {
            break;
        }
    }
    return count;
}

__global__ void test_difficulty(const uint8_t *hash, uint32_t *result)
{
    *result = difficulty(hash);
}
