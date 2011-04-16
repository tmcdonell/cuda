/* -----------------------------------------------------------------------------
 *
 * Module    : FoldSeg
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * ---------------------------------------------------------------------------*/

#include "foldSeg.h"

#include "utils.h"
#include "operator.h"


/* -----------------------------------------------------------------------------
 * Warps
 * ---------------------------------------------------------------------------*/

template <class op, typename T, bool inclusive>
__device__ T scan_warp(volatile T *xs)
{
    const unsigned int idx  = threadIdx.x;
    const unsigned int lane = idx & (WARP_SIZE-1);

    if (lane >=  1) xs[idx] = op::apply(xs[idx], xs[idx- 1]);
    if (lane >=  2) xs[idx] = op::apply(xs[idx], xs[idx- 2]);
    if (lane >=  4) xs[idx] = op::apply(xs[idx], xs[idx- 4]);
    if (lane >=  8) xs[idx] = op::apply(xs[idx], xs[idx- 8]);
    if (lane >= 16) xs[idx] = op::apply(xs[idx], xs[idx-16]);

    if (inclusive) return xs[idx];
    else           return (lane > 0) ? ptr[idx-1] : op::identity();
}

#if 0
/*
 * Intra-warp segmented (inclusive) scan. This uses a head flags array to denote
 * the start of each section
 */
template <class op, typename T>
__device__ T segscan_warp
(
    volatile T            *xs,
    volatile unsigned int *flags
)
{
    const unsigned int idx  = threadIdx.x;
    const unsigned int lane = idx & (WARP_SIZE-1);

#define DO_SEGSCAN(n)                                                          \
    if (lane >= n)                                                             \
    {                                                                          \
        xs[idx]    = flags[idx] ? xs[idx] : op::apply(xs[idx-n], xs[idx]);     \
        flags[idx] = flags[idx] | flags[idx-n];                                \
    }

    DO_SEGSCAN( 1);
    DO_SEGSCAN( 2);
    DO_SEGSCAN( 4);
    DO_SEGSCAN( 8);
    DO_SEGSCAN(16);

    return xs[idx];
}
#endif


template <class op, typename T, bool inclusive>
__device__ T segscan_warp
(
    volatile T            *xs,
    volatile unsigned int *flags
)
{
    const unsigned int idx  = threadIdx.x;
    const unsigned int lane = idx & (WARP_SIZE-1);

    /*
     * Convert head flags into minimum index form. The multiplication avoids a
     * branch by relying on the flags being boolean values.
     */
    flags[lane]        *= lane;
    unsigned int mindex = scan_warp<Max, unsigned int, true>(flags);

    /*
     * Segmented scan across warp size of 32
     */
    if (lane >= mindex +  1) { xs[idx] = op::apply(xs[idx], xs[idx- 1]); }
    if (lane >= mindex +  2) { xs[idx] = op::apply(xs[idx], xs[idx- 2]); }
    if (lane >= mindex +  4) { xs[idx] = op::apply(xs[idx], xs[idx- 4]); }
    if (lane >= mindex +  8) { xs[idx] = op::apply(xs[idx], xs[idx- 8]); }
    if (lane >= mindex + 16) { xs[idx] = op::apply(xs[idx], xs[idx-16]); }

    if (inclusive) return xs[idx];
    else           return (lane > 0 && mindex != lane) ? xs[idx-1] : op::identity();
}


/* -----------------------------------------------------------------------------
 * Blocks
 * ---------------------------------------------------------------------------*/

template <unsigned int blockSize, bool lengthIsPow2, class op, typename T>
__device__ void segfold_recursive
(
    const T             *d_xs,
    const unsigned int  *d_flags,
    T                   *d_out,
    unsigned int        *d_out_flags,
    unsigned int        length
)
{
    __shared__ T            xs[blockSize];
    __shared__ unsigned int hd[blockSize];

    const unsigned int idx       = threadIdx.x;
    const unsigned int lane      = idx & (WARP_SIZE-1);
    const unsigned int warp_id   = idx / WARP_SIZE;
    const unsigned int warp_fst  = warp_id * WARP_SIZE;
    const unsigned int warp_last = warp_fst + WARP_SIZE - 1;

    xs[idx] = 0;
    hd[idx] = 0;

    /*
     * Reduce multiple elements per thread. The actual number is determined by
     * the active number of thread blocks. More blocks results in a larger
     * `gridDim', and fewer elements per thread.
     *
     * Warps process chunks of consecutive elements. When the end of a segment
     * is reached, the (partial) result is written out. When the next chunk of
     * elements begin with an "open" segment, the last thread of the warp
     * carries over its reduction value.
     */
    unsigned int i;
    unsigned int chunk = blockSize * 2 * gridDim.x;
    unsigned int begin = warp_id * chunk + lane;

    for (i = begin; i < begin + chunk; i += WARP_SIZE)
    {
        xs[idx] = d_xs[i];
        hd[idx] = d_flags[i];
    }


    /*
     * Before overwriting the input head flags, record whether this warp begins
     * with an "open" segment.
     */
    bool warp_is_open = (d_flags[warp_fst] == 0);
}



#if 0
template <typename T>
struct foldSeg_plan
{
    T            **block_sums;
    unsigned int **block_idx;
    size_t       num_levels;
};


/*
 * Intra-warp segmented (inclusive) scan that uses an array denoting the segment
 * index each element belongs to
 */
template <class op, typename T>
__device__ T segscan_warp
(
    volatile T          *xs,
    const unsigned int  *segment_index,
)
{
    const unsigned int idx    = threadIdx.x;
    const unsigned int lane   = idx & (WARP_SIZE-1);
    const unsigned int segidx = segment_index[idx];

    if (lane >=  1 && segidx == segment_index[idx- 1]) { xs[idx] = op::apply(xs[idx], xs[idx- 1]); }
    if (lane >=  2 && segidx == segment_index[idx- 2]) { xs[idx] = op::apply(xs[idx], xs[idx- 2]); }
    if (lane >=  4 && segidx == segment_index[idx- 4]) { xs[idx] = op::apply(xs[idx], xs[idx- 4]); }
    if (lane >=  8 && segidx == segment_index[idx- 8]) { xs[idx] = op::apply(xs[idx], xs[idx- 8]); }
    if (lane >= 16 && segidx == segment_index[idx-16]) { xs[idx] = op::apply(xs[idx], xs[idx-16]); }

    return xs[idx];
}


/*
 * Segmented reduction in shared memory
 */
template <unsigned int blockSize, bool lengthIsPow2, class op, typename T>
__device__ void
foldSeg_recursive
(
    const T             *d_val,
    const unsigned int  *d_idx,
    T                   *d_out_val,
    unsigned int        *d_out_idx,
    unsigned int        length
)
{
    __shared__ T            s_val[blockSize];
    __shared__ unsigned int s_idx[blockSize];

    /*
     * Reduce multiple elements per thread. The actual number is determined by
     * the number of active thread blocks. More blocks will result in a larger
     * `gridDim', and fewer elements per thread.
     *
     * Each warp processes `gridDim' number of consecutive elements.
     */
    unsigned int i;
    const unsigned int tid      = blockSize * blockIdx.x + threadIdx.x;
    const unsigned int lane     = threadIdx.x & (WARP_SIZE - 1);
    const unsigned int warp_id  = thread_id / WARP_SIZE;

    const unsigned int interval = blockSize * 2 * gridDim.x;
    const unsigned int begin    = warp_id * interval + lane;
    const unsigned int end      = begin + interval;

    /*
     * Initialise values
     */
    s_val[threadIdx.x] = op::apply(op::identity(), d_val[begin]);
    s_idx[threadIdx.x] = d_idx[begin];

    for (i = begin; i < end; i += WARP_SIZE)
    {
        /*
         * With row indices and partial reduction in shared memory, compute a
         * segmented scan to combine values belonging to the same segment. For
         * example:
         *
         *           0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15   # thread lane
         *   s_idx [ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]  # row indices
         *   s_val [ 4, 6, 5, 0, 8, 3, 2, 8, 3, 1, 4, 9, 2, 5, 2, 4]  # values
         *
         * After the segmented scan:
         *
         *   s_val [ 4,10,15,15,23,26, 2,10,13,14, 4,13,15,20,22,26]
         */
        foldSeg_warp(s_val, s_idx, lane);

        /*
         * If this segment ends, store the result. Otherwise ...
         */
        if (lane < WARP_SIZE-1 && s_idx[threadIdx.x] != s_idx[threadIdx.x+1])
        {
            d_out_val[tid + s_idx[threadIdx.x]] = s_val[threadIdx.x];
            d_out_idx[tid + s_idx[threadIdx.x]] = s_idx[threadIdx.x];
        }
    }
}


template <class op, typename T>
static void
foldSeg_dispatch
(
    const T             *d_in,
    T                   *d_out,
    scan_plan<T>        *plan,
    int                 N,
    int                 level
)
{
}


static void
foldSeg_control
(
    int         n,
    int         &blocks,
    int         &threads,
    int         maxThreads = MAX_THREADS,
    int         maxBlocks  = MAX_BLOCKS
)
{
    threads = (n < maxThreads*2) ? ceilPow2((n+1)/2) : maxThreads;
    blocks  = (n + threads * 2 - 1) / (threads * 2);
    blocks  = min(blocks, maxBlocks);
}


/*
 * Allocate temporary memory used by the scan. At each level we need to store at
 * least (number_blocks + number_segments - 1) partial reductions. The final
 * result is produced when only a single block is launched.
 */
template <typename T>
static void
foldSeg_init(int N, foldSeg_plan<T> &plan)
{
}

template <typename T>
static void
foldSeg_finalise(foldSeg_plan<T> &plan)
{
}


template <class op, typename T>
void
foldSeg
(
    const T             *d_in,
    const unsigned int  *d_segment_idx;
    T                   *d_out,
    int                 length
)
{
}
#endif
