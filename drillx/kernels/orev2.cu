#include <stdint.h>
#include <stdio.h>
#include "orev2.h"
#include "utils.h"
#include "keccak.h"

__device__ uint32_t global_best_difficulty = 0;
__device__ unsigned long long int global_best_nonce = 0;
__device__ uint32_t device_best_difficulty; 
__device__ unsigned long long int device_best_nonce; 

__device__ size_t noise[NOISE_SIZE_BYTES / USIZE_BYTE_SIZE];

extern "C" void set_noise(const size_t *data)
{
    for (int device = 0; device < device_count; ++device)
    {
        cudaSetDevice(device);
        cudaMemcpyToSymbol(noise, data, NOISE_SIZE_BYTES, 0, cudaMemcpyHostToDevice);
    }
}

extern "C" void get_noise(size_t *host_data)
{
    for (int device = 0; device < device_count; ++device)
    {
        cudaSetDevice(device);
        cudaMemcpyFromSymbol(host_data, noise, NOISE_SIZE_BYTES, 0, cudaMemcpyDeviceToHost);
    }
}

extern "C" void drill_hash(uint8_t *challenge, uint8_t *out, uint64_t round)
{
    const uint64_t FIXED_NONCE_RANGE = 1000000000ULL; // 1 billion

    if (round == 0)
    {
        unsigned long long int zero = 0;
        uint32_t zero_difficulty = 0;

        cudaMemcpyToSymbol(global_best_nonce, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(global_best_difficulty, &zero_difficulty, sizeof(zero_difficulty), 0, cudaMemcpyHostToDevice);
    }

 // Allocate device memory for input data and results
uint8_t *d_challenge[device_count];
uint64_t *d_best_nonce[device_count];
uint32_t *d_best_difficulty[device_count];
for (int device = 0; device < device_count; ++device)
{
    cudaSetDevice(device);
    cudaError_t cuda_status;

    cuda_status = cudaMalloc((void **)&d_challenge[device], 32);
    if (cuda_status != cudaSuccess) {
        printf("cudaMalloc failed for d_challenge on GPU %d: %s\n", device, cudaGetErrorString(cuda_status));
        // Handle error
        
    }

    cuda_status = cudaMemcpy(d_challenge[device], challenge, 32, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        printf("cudaMemcpy failed for d_challenge on GPU %d: %s\n", device, cudaGetErrorString(cuda_status));
        // Handle error
        
    }

    cuda_status = cudaMalloc((void **)&d_best_nonce[device], sizeof(uint64_t));
    if (cuda_status != cudaSuccess) {
        printf("cudaMalloc failed for d_best_nonce on GPU %d: %s\n", device, cudaGetErrorString(cuda_status));
        // Handle error
        
    }

    cuda_status = cudaMalloc((void **)&d_best_difficulty[device], sizeof(uint32_t));
    if (cuda_status != cudaSuccess) {
        printf("cudaMalloc failed for d_best_difficulty on GPU %d: %s\n", device, cudaGetErrorString(cuda_status));
        // Handle error
        
    }
}

uint64_t total_stride = number_blocks * number_threads;
uint64_t nonce_per_device = FIXED_NONCE_RANGE / device_count;

for (int device = 0; device < device_count; ++device)
{
    cudaSetDevice(device);
    uint64_t start_nonce = device * nonce_per_device;
    printf("Launching kernel on GPU %d with start nonce %llu\n", device, (unsigned long long int)start_nonce);

    // Print all arguments before launching the kernel
    printf("Kernel arguments for GPU %d:\n", device);
    printf("d_challenge: %p\n", d_challenge[device]);
    printf("total_stride: %llu\n", total_stride);
    printf("round: %llu\n", round);
    printf("batch_size: %u\n", batch_size);
    printf("start_nonce: %llu\n", (unsigned long long int)start_nonce);
    printf("d_best_nonce: %p\n", d_best_nonce[device]);
    printf("d_best_difficulty: %p\n", d_best_difficulty[device]);

    cudaError_t cuda_status;
    // Launch the kernel with error checking
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("Previous CUDA error before kernel launch on GPU %d: %s\n", device, cudaGetErrorString(cuda_status));
        // Handle error
        
    }

    kernel_start_drill<<<number_blocks / device_count, number_threads>>>(d_challenge[device], total_stride, round, batch_size, start_nonce, d_best_nonce[device], d_best_difficulty[device]);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("CUDA error during kernel launch on GPU %d: %s\n", device, cudaGetErrorString(cuda_status));
        // Handle error
        
    }
}

    for (int device = 0; device < device_count; ++device)
    {
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    }

    unsigned long long int best_nonce = 0;
    uint32_t best_difficulty = 0;
    int best_device = -1;
    for (int device = 0; device < device_count; ++device)
    {
        unsigned long long int device_nonce;
        uint32_t device_difficulty;
        cudaSetDevice(device);
        cudaMemcpy(&device_nonce, d_best_nonce[device], sizeof(device_nonce), cudaMemcpyDeviceToHost);
        cudaMemcpy(&device_difficulty, d_best_difficulty[device], sizeof(device_difficulty), cudaMemcpyDeviceToHost);

        if (device_difficulty > best_difficulty)
        {
            best_difficulty = device_difficulty;
            best_nonce = device_nonce;
            best_device = device;
        }
    }

    uint64_t global_best_nonce_value;
    uint32_t global_best_difficulty_value;
    cudaMemcpyFromSymbol(&global_best_nonce_value, global_best_nonce, sizeof(global_best_nonce), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&global_best_difficulty_value, global_best_difficulty, sizeof(global_best_difficulty), 0, cudaMemcpyDeviceToHost);

    printf("Best Hash Found By Device %d\n", best_device);

    memcpy(out, &global_best_nonce_value, sizeof(global_best_nonce_value));

    for (int device = 0; device < device_count; ++device)
    {
        cudaSetDevice(device);
        cudaFree(d_challenge[device]);
        cudaFree(d_best_nonce[device]);
        cudaFree(d_best_difficulty[device]);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}


__global__ void kernel_start_drill(
    uint8_t *d_challenge,
    uint64_t stride,
    uint64_t round,
    uint32_t batch_size,
    uint64_t start_nonce,
    uint64_t *device_best_nonce,
    uint32_t *device_best_difficulty)
{
    printf("here\n");
    uint64_t iters = 0;
    uint64_t nonce = start_nonce + threadIdx.x + (blockIdx.x * blockDim.x);
    uint64_t local_best_nonce = nonce;
    uint32_t local_best_difficulty = 0;
    uint8_t result[32];

    
    while (iters < batch_size)
    {
        kernel_drill_hash(d_challenge, &nonce, result);
        uint32_t hash_difficulty = difficulty(result);
        if (hash_difficulty > local_best_difficulty)
        {
            local_best_difficulty = hash_difficulty;
            local_best_nonce = nonce;
        }
        nonce += stride; 
        iters += 1;
    }

    *device_best_difficulty = local_best_difficulty;
    *device_best_nonce = local_best_nonce;

    uint32_t prev_global_best_difficulty = atomicMax(&global_best_difficulty, local_best_difficulty);
    if (local_best_difficulty > prev_global_best_difficulty) {
        atomicExch(&global_best_nonce, local_best_nonce);
    }
}



extern "C" void single_drill_hash(uint8_t *challenge, uint64_t nonce, uint8_t *out)
{
    // Allocate device memory for input and output data
    uint8_t *d_challenge, *d_out;
    uint64_t *d_nonce;

    cudaMalloc((void **)&d_challenge, 32);
    cudaMalloc((void **)&d_out, 32);
    cudaMalloc((void **)&d_nonce, 8);

    // Copy the host data to the device
    cudaMemcpy(d_challenge, challenge, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonce, &nonce, 8, cudaMemcpyHostToDevice);

    // Launch the kernel to perform the hash operation
    single_drill_hash_routine<<<1, 1>>>(d_challenge, d_nonce, d_out);

    // Retrieve the results back to the host
    cudaMemcpy(out, d_out, 32, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_challenge);
    cudaFree(d_out);
    cudaFree(d_nonce);
}

__global__ void single_drill_hash_routine(uint8_t *d_challenge, uint64_t *nonce, uint8_t *out)
{
    kernel_drill_hash(d_challenge, nonce, out);
}

__device__ void kernel_drill_hash(uint8_t *d_challenge, uint64_t *d_nonce, uint8_t *d_out)
{
    // Initialize the state preimage: [challenge, nonce]
    uint8_t state_preimage[40];
    memcpy(state_preimage, d_challenge, 32);
    memcpy(state_preimage + 32, d_nonce, 8);

    // Calculate initial state using keccak
    uint8_t state[32];
    keccak(state_preimage, 40, state);

    // Perform the drilling and hashing operations
    drill(state);
    keccak(state, 32, d_out);
}

__device__ void mine(
    uint8_t *d_challenge,
    uint64_t nonce,
    uint32_t *local_best_difficulty,
    uint64_t *local_best_nonce)
{
    // Drillhash
    uint8_t result[32];
    kernel_drill_hash(d_challenge, &nonce, result);

    // Compute difficulty
    uint32_t d = difficulty(result);
    if (d > *local_best_difficulty)
    {
        *local_best_nonce = nonce;
        *local_best_difficulty = d;
    }
}

__device__ void drill(uint8_t *state)
{
    // Initialize r
    uint64_t r = initialize_r(state);

    for (uint64_t i = 0; i < 4; i++)
    {
        // // Fetch noise
        uint64_t idxs[8];
        indices(state, idxs);
        for (uint64_t j = 0; j < 8; j++)
        {
            state[8 * i + j] ^= do_reads(state, idxs[j], r);
        }

        // Do ops
        indices(state, idxs);
        for (uint64_t j = 0; j < OPS; j++)
        {
            r ^= op(idxs[j % 8], r, j);
        }

        // hash state
        uint8_t state_preimage[40];
        memcpy(state_preimage, state, 32);
        memcpy(state_preimage + 32, &r, 8);
        keccak(state_preimage, 40, state);
    }
}

__device__ uint64_t initialize_r(const uint8_t *state)
{
    uint8_t rbytes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint8_t c = 0;

    for (int i = 0; i < 8; i++)
    {
        rbytes[i] = state[c % 32];
        c ^= rbytes[i];
    }

    uint64_t r = 0;
    for (int i = 0; i < 8; i++)
    {
        r |= (uint64_t)rbytes[i] << (8 * i);
    }

    return r;
}

__device__ void indices(uint8_t *state, uint64_t *indices)
{
    for (int i = 0; i < 8; i++)
    {
        indices[i] = (uint64_t)state[4 * i] |
                     (uint64_t)(state[4 * i + 1]) << 8 |
                     (uint64_t)(state[4 * i + 2]) << 16 |
                     (uint64_t)(state[4 * i + 3]) << 24;
    }
}

__device__ uint8_t do_reads(uint8_t *state, uint64_t index, uint64_t r)
{
    for (int i = 0; i < READS; i++)
    {
        index ^= noise[index % NOISE_LEN] * (uint64_t)state[i % 32];
    }

    return (uint8_t)(noise[index % NOISE_LEN] >> (r % 8));
}

__device__ uint64_t op(uint64_t a, uint64_t b, uint64_t opcount)
{
    Opcode opcode = static_cast<Opcode>((opcount ^ b) % CARDINALITY);
    switch (opcode)
    {
    case Add:
        return a + b;
    case Sub:
        return a - b;
    case Mul:
        return a * b;
    case Div:
        if (a > b)
        {
            return a / saturating_add(b, 2);
        }
        else
        {
            return b / saturating_add(a, 2);
        }
    case Xor:
        return a ^ b;
    case Right:
        return a >> (b % 64);
    case Left:
        return a << (b % 64);
    }
}
