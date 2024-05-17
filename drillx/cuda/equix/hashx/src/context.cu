/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#include <stdlib.h>
#include <string.h>

#include <hashx.h>
#include "context.h"
#include "compiler.h"
#include "program.h"

#define STRINGIZE_INNER(x) #x
#define STRINGIZE(x) STRINGIZE_INNER(x)

/* Salt used when generating hash functions. Useful for domain separation. */
#ifndef HASHX_SALT
#define HASHX_SALT HashX v1
#endif

/* Blake2b params used to generate program keys */
__device__ const blake2b_param hashx_blake2_params = {
	.digest_length = 64,
	.key_length = 0,
	.fanout = 1,
	.depth = 1,
	.leaf_length = 0,
	.node_offset = 0,
	.node_depth = 0,
	.inner_length = 0,
	.reserved = { 0 },
	.salt = STRINGIZE(HASHX_SALT),
	.personal = { 0 }
};

__device__ hashx_ctx* hashx_alloc(hashx_type type) {
	if (!HASHX_COMPILER && (type & HASHX_COMPILED)) {
		return HASHX_NOTSUPP;
	}

	hashx_ctx* ctx = NULL;
    if (cudaMalloc(&ctx, sizeof(hashx_ctx)) != cudaSuccess) {
        goto failure;
    }

	ctx->code = NULL;
	if (type & HASHX_COMPILED) {
		if (!hashx_compiler_init(ctx)) {
			goto failure;
		}
		ctx->type = HASHX_COMPILED;
	}
	else {
		if (cudaMalloc(&ctx->program, sizeof(hashx_program)) != cudaSuccess) {
            goto failure;
        }
		ctx->type = HASHX_INTERPRETED;
	}
#ifdef HASHX_BLOCK_MODE
	memcpy(&ctx->params, &hashx_blake2_params, 32);
#endif
#ifndef NDEBUG
	ctx->has_program = false;
#endif
	return ctx;
failure:
	hashx_free(ctx);
	return NULL;
}

__device__ void hashx_free(hashx_ctx* ctx) {
	if (ctx != NULL && ctx != HASHX_NOTSUPP) {
		if (ctx->code != NULL) {
			if (ctx->type & HASHX_COMPILED) {
				hashx_compiler_destroy(ctx);
			}
			else {
				cudaFree(ctx->program);
			}
		}
		cudaFree(ctx);
	}
}
