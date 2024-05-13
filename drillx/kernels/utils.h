#ifndef UTILS_H
#define UTILS_H

extern int number_multi_processors;
extern int number_blocks;
extern int number_threads;
extern int max_threads_per_mp;
extern int batch_size;

// Initializes gpu parameters
extern "C" void gpu_init(uint32_t batchsize, uint32_t threads_per_block);

// Greatest common denominator
// Used in gpu_init() to calculate block_size
int gcd(int a, int b);

// Saturating add
__device__ uint64_t saturating_add(uint64_t a, uint64_t b);

// Hash difficulty
__device__ uint32_t difficulty(const uint8_t* hash);


#endif
