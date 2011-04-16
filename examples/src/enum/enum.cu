/* -----------------------------------------------------------------------------
 *
 * Module    : Enum
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * ---------------------------------------------------------------------------*/

#include "enum.h"

#include "algorithms.h"
#include "utils.h"
#include <stdint.h>


static void
enum_control(int32_t n, uint32_t &blocks, uint32_t &threads)
{
    threads = min(ceilPow2(n), MAX_THREADS);
    blocks  = (n + threads - 1) / threads;
}


template <typename T, bool increasing>
__global__ static void
enum_core(T *out, const T from, const T then, const T to)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const T val  = from + (then-from) * idx;

    if (increasing) if (val <= to) out[idx] = val;
    else            if (val >= to) out[idx] = val;
}

template <typename T>
static void
enumFromThenTo(T *out, const T from, const T then, const T to)
{
    uint32_t threads;
    uint32_t blocks;
    int32_t  n          = 1 + (to - from) / (then - from);
    bool     increasing = then-from > 0;

    if (n <= 0) return;

    enum_control(n, blocks, threads);
    if (increasing) enum_core<T,true> <<<blocks,threads>>>(out, from, then, to);
    else            enum_core<T,false><<<blocks,threads>>>(out, from, then, to);
}


// -----------------------------------------------------------------------------
// Instances
// -----------------------------------------------------------------------------

void enumFromTo_i(int32_t *out, int32_t l, int32_t u)
{
    enumFromThenTo<int32_t>(out, l, l+1, u);
}

