#ifndef KECCAK_H
#define KECCAK_H

__device__ void keccak(const uint8_t *message, int message_len, uint8_t *output);

#endif
