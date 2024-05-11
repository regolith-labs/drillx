#ifndef OREV2_H
#define OREV2_H

#define NOISE_SIZE_BYTES 1000 * 1000
#define USIZE_BYTE_SIZE 8
#define NOISE_LEN (NOISE_SIZE_BYTES / USIZE_BYTE_SIZE)
#define READS 256
#define OPS 64
#define CARDINALITY 7

extern __device__ uint64_t noise[NOISE_SIZE_BYTES / USIZE_BYTE_SIZE];

extern "C" void set_noise(const uint64_t *data);
extern "C" void get_noise(uint64_t *data);
// host function that calls kernel_start_drill, a kernel that initializes the parallel routine that mines hashes for some time,
extern "C" void drill_hash(uint8_t *challenge, uint8_t *out, uint64_t secs);
__global__ void kernel_start_drill(uint8_t *d_challenge, uint8_t *d_result, uint64_t stride, unsigned long long int target_cycles);
__device__ void mine(uint8_t *d_challenge, uint64_t nonce, uint32_t *local_best_difficulty, uint64_t *local_best_nonce);

// host function that calls kernel_single_drill_hash, a non-parallel routine that does one hash
extern "C" void single_drill_hash(uint8_t *d_challenge, uint64_t nonce, uint8_t *out);
__global__ void single_drill_hash_routine(uint8_t *d_challenge, uint64_t *nonce, uint8_t *out);

// The inner function called by both the parallel and non-parallel routine that does one hash
__device__ void kernel_drill_hash(uint8_t *d_state_preimage, uint64_t *d_nonce, uint8_t *d_out);

// drill and its subrountines
__device__ void drill(uint8_t *state);
__device__ uint64_t initialize_r(const uint8_t *state);
__device__ void indices(uint8_t *state, uint64_t *indices);
__device__ uint8_t do_reads(uint8_t *state, uint64_t index, uint64_t r);
__device__ uint64_t op(uint64_t a, uint64_t b, uint64_t j);

enum Opcode
{
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Xor = 4,
    Right = 5,
    Left = 6
};

#endif
