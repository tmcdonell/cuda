/*
 * Module    : Prelude
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 */

#include "utils.h"
#include "kernels.h"
#include "operator.h"
#include "cudpp/cudpp_globals.h"


/*
 * Combine two arrays using the given binary operator function. A single thread
 * is used to compute each result pair.
 */
template <class op, bool lengthIsPow2, typename Ta, typename Tb, typename Tc>
__global__ static void
zipWith_core
(
    Ta  *xs,
    Tb  *ys,
    Tc  *out,
    int length
)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (lengthIsPow2 || idx < length)
        out[idx] = op::apply(xs[idx], ys[idx]);
}


template <class op, typename Ta, typename Tb, typename Tc>
void
zipWith
(
    Ta  *xs,
    Tb  *ys,
    Tc  *zs,
    int length
)
{
    unsigned int threads = min(ceilPow2(length), CTA_SIZE);
    unsigned int blocks  = (length + threads - 1) / threads;

    if (isPow2(length))
        zipWith_core< op,true ><<<blocks,threads>>>(xs, ys, zs, length);
    else
        zipWith_core< op,false ><<<blocks,threads>>>(xs, ys, zs, length);
}


// -----------------------------------------------------------------------------
// Instances
// -----------------------------------------------------------------------------

#if 0
void zipWith_plusf(float *xs, float *ys, float *zs, int N)
{
    zipWith< Plus<float> >(xs, ys, zs, N);
}

void zipWith_timesf(float *xs, float *ys, float *zs, int N)
{
    zipWith< Times<float> >(xs, ys, zs, N);
}
#endif

void zipWith_timesif(int *xs, float *ys, float *zs, int N)
{
    zipWith< Times<int,float,float> >(xs, ys, zs, N);
}

