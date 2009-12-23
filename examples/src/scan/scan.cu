/* -----------------------------------------------------------------------------
 *
 * Module    : Scan
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * ---------------------------------------------------------------------------*/

#include "scan.h"

#include "utils.h"
#include "operator.h"
#include "cudpp/cudpp_globals.h"
#include "cudpp/scan_kernel.cu"
#include "cudpp/vector_kernel.cu"

template <typename T>
struct scan_plan
{
    T           **block_sums;
    size_t      num_levels;
};

static inline unsigned int
calc_num_blocks(unsigned int N)
{
    return max(1u, (unsigned int)ceil((double)N / (SCAN_ELTS_PER_THREAD * CTA_SIZE)));
}


/*
 * This is the CPU-side workhorse of the scan operation, invoking the kernel on
 * each of the reduction blocks.
 */
template <class op, typename T, bool backward, bool exclusive>
static void
scan_recursive
(
    const T             *in,
    T                   *out,
    scan_plan<T>        *plan,
    int                 N,
    int                 level
)
{
    size_t      num_blocks = calc_num_blocks(N);
    bool        is_full    = N == num_blocks * SCAN_ELTS_PER_THREAD * CTA_SIZE;

    dim3        grid(num_blocks, 1, 1);
    dim3        block(CTA_SIZE, 1, 1);
    size_t      smem = sizeof(T) * CTA_SIZE * 2;

#define MULTIBLOCK      0x01
#define FULLBLOCK       0x04
    int traits = 0;
    if (num_blocks > 1) traits |= MULTIBLOCK;
    if (is_full)        traits |= FULLBLOCK;

    /*
     * Set up execution parameters, and execute the scan
     */
    switch (traits)
    {
    case 0:
        scan4
            < T, ScanTraits<T, op, backward, exclusive, false, false, false> >
            <<<grid, block, smem>>>(out, in, NULL, N, 1, 1);
        break;

    case MULTIBLOCK:
        scan4
            < T, ScanTraits<T, op, backward, exclusive, false, true, false> >
            <<<grid, block, smem>>>(out, in, plan->block_sums[level], N, 1, 1);
        break;

    case FULLBLOCK:
        scan4
            < T, ScanTraits<T, op, backward, exclusive, false, false, true> >
            <<<grid, block, smem>>>(out, in, NULL, N, 1, 1);
        break;

    case MULTIBLOCK | FULLBLOCK:
        scan4
            < T, ScanTraits<T, op, backward, exclusive, false, true, true> >
            <<<grid, block, smem>>>(out, in, plan->block_sums[level], N, 1, 1);
        break;

    default:
        assert(!"Non-exhaustive patterns in match");
    }

    /*
     * After scanning the sub-blocks, we now need to combine those results by
     * taking the last value from each sub-block, and adding that to each of the
     * successive blocks (i.e. scan across the sub-computations)
     */
    if (num_blocks > 1)
    {
        T *sums = plan->block_sums[level];

        scan_recursive
            <op, T, backward, true>
            (sums, sums, plan, num_blocks, level+1);

        vectorAddUniform4
            <T, op, SCAN_ELTS_PER_THREAD>
            <<<grid,block>>>
            (out, sums, N, 4, 4, 0, 0);
    }

#undef MULTIBLOCK
#undef FULLBLOCK
}


/*
 * Allocate temporary memory used by the scan.
 */
template <typename T>
static void
scan_init(int N, scan_plan<T> *plan)
{
    size_t level        = 0;
    size_t elements     = N;
    size_t num_blocks;

    /*
     * Determine how many intermediate block-level summations will be required
     */
    for (elements = N; elements > 1; elements = num_blocks)
    {
        num_blocks = calc_num_blocks(elements);

        if (num_blocks > 1)
            ++level;
    }

    plan->block_sums = (T**) malloc(level * sizeof(T*));
    plan->num_levels = level;

    /*
     * Now, allocate the necessary storage at each level
     */
    for (elements = N, level = 0; elements > 1; elements = num_blocks, level++)
    {
        num_blocks = calc_num_blocks(elements);

        if (num_blocks > 1)
            cudaMalloc((void**) &plan->block_sums[level], num_blocks * sizeof(T));
    }
}


/*
 * Clean up temporary memory used by the scan
 */
template <typename T>
static void
scan_finalise(scan_plan<T> *p)
{
    for (size_t l = 0; l < p->num_levels; ++l)
        cudaFree(p->block_sums[l]);

    free(p->block_sums);
}


/*
 * Apply a binary operator to an array similar to `fold', but return a
 * successive list of values reduced from the left. The reduction will take
 * place in parallel, so the operator must be associative.
 */
template <class op, typename T, bool backward, bool exclusive>
void
scan
(
    const T     *in,
    T           *out,
    int         length
)
{
    scan_plan<T> plan;
    scan_init<T>(length, &plan);

    scan_recursive<op, T, backward, exclusive>(in, out, &plan, length, 0);

    scan_finalise<T>(&plan);
}


// -----------------------------------------------------------------------------
// Instances
// -----------------------------------------------------------------------------

void scanl_plusf(float *in, float *out, int N)
{
    scan< Plus<float>, float, false, true >(in, out, N);
}

void scanl1_plusf(float *in, float *out, int N)
{
    scan< Plus<float>, float, false, false >(in, out, N);
}

