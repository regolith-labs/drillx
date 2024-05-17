/* Copyright (c) 2020 tevador <tevador@gmail.com> */
/* See LICENSE for licensing information */

#ifndef COMPILER_H
#define COMPILER_H

#include <stdint.h>
#include <stdbool.h>
#include <hashx.h>
#include "program.h"

#define HASHX_COMPILER 0
#define hashx_compile

HASHX_PRIVATE bool hashx_compiler_init(hashx_ctx* compiler);
HASHX_PRIVATE void hashx_compiler_destroy(hashx_ctx* compiler);

#define ALIGN_SIZE(pos, align) ((((pos) - 1) / (align) + 1) * (align))

#define COMP_PAGE_SIZE 4096
#define COMP_RESERVE_SIZE 1024
#define COMP_AVG_INSTR_SIZE 5
#define COMP_CODE_SIZE                                                        \
	ALIGN_SIZE(                                                               \
		HASHX_PROGRAM_MAX_SIZE * COMP_AVG_INSTR_SIZE + COMP_RESERVE_SIZE,     \
	COMP_PAGE_SIZE)

#endif
