/*
 * Name      : VectorAdd
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * Element-wise addition of two (floating-point) vectors
 */


extern "C" __global__ void VecAdd(float *xs, float *ys, float *out, int N)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N)
        out[idx] = xs[idx] + ys[idx];
}

