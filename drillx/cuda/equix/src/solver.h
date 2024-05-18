/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#ifndef SOLVER_H
#define SOLVER_H

#include <equix.h>
#include <hashx_endian.h>
#include <stdbool.h>
#include "context.h"

#define EQUIX_STAGE1_MASK ((1ull << 15) - 1)
#define EQUIX_STAGE2_MASK ((1ull << 30) - 1)
#define EQUIX_FULL_MASK ((1ull << 60) - 1)


static inline bool tree_cmp1(const equix_idx* left, const equix_idx* right) {
	return *left <= *right;
}

static inline bool tree_cmp2(const equix_idx* left, const equix_idx* right) {
	return load32(left) <= load32(right);
}

static inline bool tree_cmp4(const equix_idx* left, const equix_idx* right) {
	return load64(left) <= load64(right);
}

__device__ void hash_stage0i(hashx_ctx* hash_func, uint64_t* out, uint32_t i);

uint32_t equix_solver_solve(uint64_t* hashes, solver_heap* heap, equix_solution output[EQUIX_MAX_SOLS]);


#endif
