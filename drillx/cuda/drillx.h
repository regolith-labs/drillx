#ifndef DRILLX_H
#define DRILLX_H

#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "hashx/src/context.h"

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint8_t *out);

// __global__ void do_hash_stage0i(hashx_ctx* hash_func, uint8_t* out);
__global__ void do_hash_stage0i(hashx_ctx* ctxs[BATCH_SIZE], uint64_t* hash_space[INDEX_SPACE]);

#endif
