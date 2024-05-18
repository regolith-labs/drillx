#ifndef DRILLX_H
#define DRILLX_H

#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

#define BATCH_SIZE 8

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out);

__global__ void do_hash_stage0i(hashx_ctx* ctxs[BATCH_SIZE], uint64_t* hash_space);

#endif
