/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

#include "stubs.h"
#include "kernels.h"

#define NUM_BANKS       16
#define LOG_NUM_BANKS   4

// Define this to more rigorously avoid bank conflicts, even at the lower (root) levels of the tree
//#define ZERO_BANK_CONFLICTS 

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#ifdef CHECK_BANK_CONFLICTS
#define TEMP(index)   cutilBankChecker(temp, index)
#else
#define TEMP(index)   temp[index]
#endif

///////////////////////////////////////////////////////////////////////////////
// Work-efficient compute implementation of scan, one thread per 2 elements
// Work-efficient: O(log(n)) steps, and O(n) adds.
// Also shared storage efficient: Uses n + n/NUM_BANKS shared memory -- no ping-ponging
// Also avoids most bank conflicts using single-element offsets every NUM_BANKS elements.
//
// In addition, If ZERO_BANK_CONFLICTS is defined, uses 
//     n + n/NUM_BANKS + n/(NUM_BANKS*NUM_BANKS) 
// shared memory. If ZERO_BANK_CONFLICTS is defined, avoids ALL bank conflicts using 
// single-element offsets every NUM_BANKS elements, plus additional single-element offsets 
// after every NUM_BANKS^2 elements.
//
// Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums 
// and Their Applications", or Prins and Chatterjee PRAM course notes:
// https://www.cs.unc.edu/~prins/Classes/633/Handouts/pram.pdf
// 
// This work-efficient version is based on the algorithm presented in Guy Blelloch's
// Excellent paper "Prefix sums and their applications".
// http://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf
//
// Pro: Work Efficient, very few bank conflicts (or zero if ZERO_BANK_CONFLICTS is defined)
// Con: More instructions to compute bank-conflict-free shared memory addressing,
// and slightly more shared memory storage used.
//
// @param g_odata  output data in global memory
// @param g_idata  input data in global memory
// @param n        input number of elements to scan from input data
__global__ void scan_best(float *g_odata, float *g_idata, int n)
{
    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  float temp[];

    int thid = threadIdx.x;

    int ai = thid;
    int bi = thid + (n/2);

    // compute spacing to avoid bank conflicts
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Cache the computational window in shared memory
    TEMP(ai + bankOffsetA) = g_idata[ai]; 
    TEMP(bi + bankOffsetB) = g_idata[bi]; 

    int offset = 1;

    // build the sum in place up the tree
    for (int d = n/2; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            TEMP(bi) += TEMP(ai);
        }

        offset *= 2;
    }

    // scan back down the tree

    // clear the last element
    if (thid == 0)
    {
        int index = n - 1;
        index += CONFLICT_FREE_OFFSET(index);
        TEMP(index) = 0;
    }   

    // traverse down the tree building the scan in place
    for (int d = 1; d < n; d *= 2)
    {
        offset /= 2;

        __syncthreads();

        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t  = TEMP(ai);
            TEMP(ai) = TEMP(bi);
            TEMP(bi) += t;
        }
    }

    __syncthreads();

    // write results to global memory
    g_odata[ai] = TEMP(ai + bankOffsetA); 
    g_odata[bi] = TEMP(bi + bankOffsetB); 
}


// Wrapper to be called from Haskell
void plus_scan(float *out, float *in, int n)
{
    dim3   threads(n,1,1);
    dim3   grid(256,1,1);

    plus_scan_kernel<<<grid, threads>>>(out, in, n);
}

