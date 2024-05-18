/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#include <stdlib.h>
#include <equix.h>
#include "context.h"
#include "solver_heap.h"

// TODO hash_func context can be removed from here

equix_ctx* equix_alloc(equix_ctx_flags flags) {
	equix_ctx* ctx_failure = NULL;
	equix_ctx* ctx = NULL;
	
	// Allocate unified memory for equix_ctx
  if (cudaMallocManaged(&ctx, sizeof(equix_ctx)) != cudaSuccess) {
      goto failure;
  }
	
	ctx->flags = (equix_ctx_flags)(flags & EQUIX_CTX_COMPILE);
	ctx->hash_func = hashx_alloc(flags & EQUIX_CTX_COMPILE ?
		HASHX_COMPILED : HASHX_INTERPRETED);
	if (ctx->hash_func == NULL) {
		goto failure;
	}
	if (ctx->hash_func == HASHX_NOTSUPP) {
		ctx_failure = EQUIX_NOTSUPP;
		goto failure;
	}
	if (flags & EQUIX_CTX_SOLVE) {
		if (cudaMallocManaged(&ctx->heap, sizeof(solver_heap)) != cudaSuccess) {
    	goto failure;
    }
	}
	ctx->flags = flags;
	return ctx;
failure:
	equix_free(ctx);
	return ctx_failure;
}

void equix_free(equix_ctx* ctx) {
	if (ctx != NULL && ctx != EQUIX_NOTSUPP) {
		if (ctx->flags & EQUIX_CTX_SOLVE) {
			cudaFree(ctx->heap);
		}
		hashx_free(ctx->hash_func);
		cudaFree(ctx);
	}
}