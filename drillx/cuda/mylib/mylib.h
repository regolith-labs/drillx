#ifndef MYLIB_H
#define MYLIB_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * Adds two integers.
 *
 * @param a First integer.
 * @param b Second integer.
 * @return The sum of a and b.
 */
int32_t add(int32_t a, int32_t b);


#ifdef __cplusplus
}
#endif

#endif // MYLIB_H