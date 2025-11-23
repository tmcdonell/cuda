/* -----------------------------------------------------------------------------
 *
 * Module    : Replicate
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * ---------------------------------------------------------------------------*/

#include "utils.h"
#include "replicate.h"
#include "algorithms.h"

#include <stdint.h>

static void
replicate_control(uint32_t n, uint32_t &blocks, uint32_t &threads)
{
    threads = min(ceilPow2(n), MAX_THREADS);
    blocks  = (n + threads - 1) / threads;
}


/*
 * Apply a function each element of an array. A single thread is used to compute
 * each result.
 */
template <bool lengthIsPow2>
__global__ static void
replicate_core
(
    uint32_t            *d_out,
    const uint32_t      symbol,
    const uint32_t      length
)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (lengthIsPow2 || idx < length)
	d_out[idx] = symbol;
}


/*
 * TODO: Generalise to 8- and 16-bit values.
 */
void
replicate
(
    void                *d_out,
    const uint32_t      symbol,
    const uint32_t      length
)
{
    uint32_t threads;
    uint32_t blocks;

    replicate_control(length, blocks, threads);

    if (isPow2(length)) replicate_core<true> <<<blocks,threads>>>((uint32_t*) d_out, symbol, length);
    else                replicate_core<false><<<blocks,threads>>>((uint32_t*) d_out, symbol, length);
}

