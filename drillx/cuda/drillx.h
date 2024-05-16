#ifndef DRILLX_H
#define DRILLX_H

extern "C" void hash(uint8_t *challenge, uint8_t *nonce, uint8_t *out);

__global__ void do_hash(uint8_t *d_challenge, uint8_t *d_nonce, uint8_t *d_out);

#endif
