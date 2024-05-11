#include <stdint.h>
#include <stdio.h>
#include "orev2.h"
#include "utils.h"
#include "keccak.h"

__device__ uint32_t global_best_difficulty = 0;
__device__ unsigned long long int global_best_nonce = 0;
__device__ unsigned long long int iters = 0;
__device__ unsigned long long int lock = 0;

// Define the static array globally
__device__ size_t noise[NOISE_SIZE_BYTES / USIZE_BYTE_SIZE];

// Function to set the static array's contents from the host
extern "C" void set_noise(const size_t *data)
{
    cudaMemcpyToSymbol(noise, data, NOISE_SIZE_BYTES, 0, cudaMemcpyHostToDevice);
}

// Function to read the static array's contents from the host.
// Mostly just for development (to ensure it is set properly).
extern "C" void get_noise(size_t *host_data)
{
    cudaMemcpyFromSymbol(host_data, noise, NOISE_SIZE_BYTES, 0, cudaMemcpyDeviceToHost);
}

extern "C" void drill_hash(uint8_t *challenge, uint8_t *out, uint64_t secs)
{
    // Allocate device memory for input and output data
    uint8_t *d_challenge, *d_out;
    uint64_t *d_nonce;

    cudaMalloc((void **)&d_challenge, 32);
    cudaMalloc((void **)&d_out, 32);
    cudaMalloc((void **)&d_nonce, 8);

    // Copy the host data to the device
    cudaMemcpy(d_challenge, challenge, 32, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_nonce, &nonce, 8, cudaMemcpyHostToDevice);

    // Calculate target cycle time. clockRate is in kHz
    unsigned long long int target_cycles = clock_rate * 1000 * secs;
    printf("clockrate %llu target cycles %llu\n", clock_rate, target_cycles);

    // Launch the kernel to perform the hash operation
    printf("num blocks %d threads %d\n", number_blocks, number_threads);
    uint64_t stride = number_blocks * number_threads;
    kernel_start_drill<<<number_blocks, number_threads>>>(d_challenge, d_out, stride, target_cycles);

    uint32_t host_gbd = 0;
    unsigned long long int host_iters = 0;
    cudaMemcpyFromSymbol(&host_gbd, global_best_difficulty, sizeof(host_gbd), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_iters, iters, sizeof(host_iters), 0, cudaMemcpyDeviceToHost);

    printf("best difficulty %u in %lld iters\n", host_gbd, host_iters);
    cudaMemcpy(out, &d_out, 32, cudaMemcpyDeviceToHost);

    // Retrieve the results back to the host
    cudaMemcpyFromSymbol(out, global_best_nonce, sizeof(global_best_nonce), 0, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_challenge);
    cudaFree(d_out);
    cudaFree(d_nonce);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void kernel_start_drill(
    uint8_t *d_challenge,
    uint8_t *d_result,
    uint64_t stride,
    unsigned long long int target_cycles)
{

    unsigned long long int start_cycles = clock64();
    unsigned long long int elapsed_cycles = 0;
    uint64_t nonce = threadIdx.x + (blockIdx.x * blockDim.x);
    uint64_t local_best_nonce = nonce;
    uint32_t local_best_difficulty = 0;
    uint8_t result[32];
    while (elapsed_cycles < target_cycles)
    {
        kernel_drill_hash(d_challenge, &nonce, result);
        uint32_t hash_difficulty = difficulty(result);
        memcpy(d_result, &local_best_difficulty, 4);
        if (hash_difficulty > local_best_difficulty)
        {
            local_best_difficulty = hash_difficulty;
            local_best_nonce = nonce;
        }

        nonce += stride;

        // Update elapsed time
        elapsed_cycles = clock64() - start_cycles;

        atomicAdd(&iters, 1);
    }

    // take lock
    while (!atomicMax(&lock, 1))
    {
    }
    if (local_best_difficulty >= global_best_difficulty)
    {
        global_best_difficulty = local_best_difficulty;
        global_best_nonce = local_best_nonce;
    }
    // release lock
    atomicMin(&lock, 0);
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
