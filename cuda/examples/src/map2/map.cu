/* -----------------------------------------------------------------------------
 *
 * Module    : Map
 * Copyright : (c) 2010 Trevor L. McDonell
 * License   : BSD
 *
 * ---------------------------------------------------------------------------*/


typedef unsigned int Ix;

typedef int TyOut;
typedef int TyIn0;
typedef int TyIn1;

typedef TyOut* ArrOut;

typedef struct {
    TyIn0 a;
    TyIn1 b;
} ArrElem0;

typedef struct {
    const TyIn0 *a;
    const TyIn1 *b;
} ArrIn0;


static inline __device__
ArrElem0 indexArray0(const ArrIn0 arr, const Ix idx)
{
    ArrElem0 x = { arr.a[idx], arr.b[idx] };
    return x;
}

__device__ static TyOut
apply(const ArrElem0 in0)
{
    return in0.a + in0.b;
}

/*
 * Apply the function to each element of the array. Each thread processes
 * multiple elements, striding the array by the grid size.
 */
extern "C"
__global__ void
map
(
    ArrOut              d_out,
    const ArrIn0        d_in0,
    const unsigned int  length
)
{
    unsigned int       idx;
    const unsigned int gridSize = __umul24(blockDim.x, gridDim.x);

    for (idx = __umul24(blockDim.x, blockIdx.x) + threadIdx.x; idx < length; idx += gridSize)
    {
        d_out[idx] = apply(indexArray0(d_in0, idx));
    }
}

// vim:filetype=cuda.c

