/*
 * Name      : VectorAdd
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * Element-wise addition of two (floating-point) vectors
 */


extern "C"
__global__ void VecAdd(const float *xs, const float *ys, float *out, const unsigned int N)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N)
        out[idx] = xs[idx] + ys[idx];
}

