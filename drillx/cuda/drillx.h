#ifndef DRILLX_H
#define DRILLX_H

#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "equix/src/solver_heap.h"
#include "hashx/src/context.h"

extern "C" const int BATCH_SIZE;

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint64_t *out);

extern "C" void solve_all_stages(uint64_t *hashes, uint8_t *out, uint32_t *sols);

__global__ void do_hash_stage0i(hashx_ctx** ctxs, uint64_t** hash_space);

#endif
