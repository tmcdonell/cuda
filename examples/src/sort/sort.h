/* -----------------------------------------------------------------------------
 *
 * Module    : Sort
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * ---------------------------------------------------------------------------*/

#ifndef __SORT_PRIV_H__
#define __SORT_PRIV_H__

#ifdef __cplusplus
extern "C" {
#endif

void sort_f(float *d_keys, void *d_vals, unsigned int length);
void sort_ui(unsigned int *d_keys, void *d_vals, unsigned int length);

#ifdef __cplusplus
}
#endif
#endif

