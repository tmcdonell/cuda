/* -----------------------------------------------------------------------------
 *
 * Module    : Fold
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * ---------------------------------------------------------------------------*/

#ifndef __FOLD_H__
#define __FOLD_H__

/*
 * Optimised for Tesla.
 * Maximum thread occupancy for your card may be achieved with different values.
 */
#define MAX_THREADS     128
#define MAX_BLOCKS      64

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Instances
 */
float fold_plusf(float *xs, int N);


#ifdef __cplusplus
}
#endif
#endif
