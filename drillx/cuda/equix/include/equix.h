/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#ifndef EQUIX_H
#define EQUIX_H

#include <stdint.h>
#include <stddef.h>

/*
 * The solver will return at most this many solutions.
 */
#define EQUIX_MAX_SOLS 8

/*
 * The number of indices.
 */
#define EQUIX_NUM_IDX 8

/*
 * 16-bit index.
 */
typedef uint16_t equix_idx;

/*
 *  The solution.
 */
typedef struct equix_solution {
    equix_idx idx[EQUIX_NUM_IDX];
} equix_solution;

/*
 * Opaque struct that holds the Equi-X context
 */
typedef struct equix_ctx equix_ctx;

/*
 * Flags for context creation
*/
typedef enum equix_ctx_flags {
    EQUIX_CTX_VERIFY = 0,       /* Context for verification */
    EQUIX_CTX_SOLVE = 1,        /* Context for solving */
    EQUIX_CTX_COMPILE = 2,      /* Compile internal hash function */
} equix_ctx_flags;

/* Sentinel value used to indicate unsupported type */
#define EQUIX_NOTSUPP ((equix_ctx*)-1)

/* Shared/static library definitions */
#ifdef EQUIX_SHARED
    #define EQUIX_API __attribute__ ((visibility ("default")))
#else
    #define EQUIX_API __attribute__ ((visibility ("hidden")))
#endif
#define EQUIX_PRIVATE __attribute__ ((visibility ("hidden")))

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Allocate an Equi-X context.
 *
 * @param flags is the type of context to be created
 *
 * @return pointer to a newly created context. Returns NULL on memory
 *         allocation failure and EQUIX_NOTSUPP if the requested type
 *         is not supported.
 */
EQUIX_API equix_ctx* equix_alloc(equix_ctx_flags flags);

/*
* Free an Equi-X a context.
*
* @param ctx is a pointer to the context
*/
EQUIX_API void equix_free(equix_ctx* ctx);


#ifdef __cplusplus
}
#endif

#endif
