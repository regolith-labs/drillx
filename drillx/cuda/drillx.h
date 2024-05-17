#ifndef DRILLX_H
#define DRILLX_H

#include "equix.h"
#include "hashx.h"
#include "equix/src/context.h"
#include "hashx/src/context.h"

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint8_t *out);

__global__ void do_solve_stage0(hashx_ctx* hash_func, solver_heap* heap);

#endif
