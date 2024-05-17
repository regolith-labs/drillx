/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#include <stdlib.h>
#include <stdbool.h>

#include <equix.h>
#include <hashx.h>
#include "context.h"
#include "solver.h"
#include <hashx_endian.h>

// __device__ int equix_solve(
// 	equix_ctx* ctx,
// 	const void* challenge,
// 	size_t challenge_size,
// 	equix_solution output[EQUIX_MAX_SOLS])
// {
// 	if ((ctx->flags & EQUIX_CTX_SOLVE) == 0) {
// 		return 0;
// 	}

// 	if (!hashx_make(ctx->hash_func, challenge, challenge_size)) {
// 		return 0;
// 	}

// 	return equix_solver_solve(ctx->hash_func, ctx->heap, output);
// }
