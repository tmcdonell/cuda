/*
 * Name      : zipWithPlus
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * Element-wise addition of two (floating-point) vectors
 */


__global__ void zipWithPlus(float *xs, float *ys, float *out, int length)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < length)
        out[idx] = xs[idx] + ys[idx];
}

