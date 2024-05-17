#ifndef DRILLX_H
#define DRILLX_H

#include "equix.h"
#include "hashx.h"

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint8_t *out);

// __global__ void do_hash(uint8_t *d_challenge, uint8_t *d_nonce, uint8_t *d_out);
__global__ void do_solve_stage0(hashx_ctx* hash_func, solver_heap* heap);

#endif
