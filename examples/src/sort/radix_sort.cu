/* -----------------------------------------------------------------------------
 *
 * Module    : Sort
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 *----------------------------------------------------------------------------*/

#include <cudpp.h>

#include "sort.h"


template <typename T> CUDPPDatatype static getType();
template <> CUDPPDatatype getType<float>()        { return CUDPP_FLOAT; }
template <> CUDPPDatatype getType<unsigned int>() { return CUDPP_UINT;  }


/*
 * In-place radix sort of values or key-value pairs. Values can be any 32-bit
 * type, as their payload is never inspected or manipulated.
 */
template <typename T>
static void
radix_sort
(
    unsigned int        length,
    T                   *d_keys,
    void                *d_vals = NULL,
    int                 bits    = 8 * sizeof(T)
)
{
    CUDPPHandle         plan;
    CUDPPConfiguration  cp;

    cp.datatype  = getType<T>();
    cp.algorithm = CUDPP_SORT_RADIX;
    cp.options   = (d_vals != NULL) ? CUDPP_OPTION_KEY_VALUE_PAIRS
                                    : CUDPP_OPTION_KEYS_ONLY;

    cudppPlan(&plan, cp, length, 1, 0);
    cudppSort(plan, d_keys, d_vals, bits, length);

    cudppDestroyPlan(plan);
}


/* -----------------------------------------------------------------------------
 * Instances
 * ---------------------------------------------------------------------------*/

void sort_f(float *d_keys, void *d_vals, unsigned int length)
{
    radix_sort<float>(length, d_keys, d_vals);
}

void sort_ui(unsigned int *d_keys, void *d_vals, unsigned int length)
{
    radix_sort<unsigned int>(length, d_keys, d_vals);
}

