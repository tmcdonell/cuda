/* -----------------------------------------------------------------------------
 *
 * Module    : Utils
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * ---------------------------------------------------------------------------*/


#ifndef __UTILS_H__
#define __UTILS_H__

#include <math.h>
#include <stdio.h>


#ifndef _DEBUG
#define assert(e)               ((void)0)
#else

/*
 * Test the given expression, and abort the program if it evaluates to false.
 * Only available in debug mode.
 */
#define assert(e)  \
    ((void) ((e) ? (void(0)) : __assert (#e, __FILE__, __LINE__)))
#define __assert(e, file, line) \
    ((void) fprintf (stderr, "%s:%u: failed assertion `%s'\n", file, line, e), abort())
#endif


/*
 * Macro to insert __syncthreads() in device emulation mode
 */
#ifdef __DEVICE_EMULATION__
#define __EMUSYNC               __syncthreads()
#else
#define __EMUSYNC
#endif


/*
 * Check the return status of CUDA API calls, and abort on failure. Only
 * available in debug mode.
 */
#ifndef _DEBUG
# define CUDA_SAFE_CALL_NO_SYNC(x)
# define CUDA_SAFE_CALL(x)
#else

# define CUDA_SAFE_CALL_NO_SYNC(call)                                          \
    do {                                                                       \
        cudaError err = call;                                                  \
        if(cudaSuccess != err) {                                               \
            __assert(cudaGetErrorString(err), __FILE__, __LINE__);             \
        }                                                                      \
    } while (0)

# define CUDA_SAFE_CALL(call)                                                  \
    do {                                                                       \
        CUDA_SAFE_CALL_NO_SYNC(call);                                          \
        CUDA_SAFE_CALL_NO_SYNC(cudaThreadSynchronize());                       \
    } while (0)

#endif


#ifdef __cplusplus
extern "C" {
#endif

/*
 * Determine if the input is a power of two
 */
inline bool
isPow2(unsigned int x)
{
    return ((x&(x-1)) == 0);
}


/*
 * Compute the next highest power of two
 */
inline unsigned int
ceilPow2(unsigned int x)
{
#if 0
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
#endif

    return (isPow2(x)) ? x : 1u << (int) ceil(log2((double)x));
}


/*
 * Compute the next lowest power of two
 */
inline unsigned int
floorPow2(unsigned int x)
{
#if 0
    float nf = (float) n;
    return 1 << (((*(int*)&nf) >> 23) - 127);
#endif

    int exp;
    frexp(x, &exp);
    return 1 << (exp - 1);
}

#ifdef __cplusplus
}
#endif
#endif

