/*
 * cuda_c2hs_compat.h — suppress C11 keywords that c2hs 0.28.x cannot parse.
 *
 * c2hs uses an internal C parser that predates C11 (released 2017-11-25).
 * CUDA 12+ headers use _Alignas, _Static_assert, and _Noreturn in live code
 * paths that are not guarded by __cplusplus or __STDC_VERSION__ checks visible
 * to c2hs's preprocessor.  Include this file before any CUDA system header.
 */
#ifndef CUDA_C2HS_COMPAT_H
#define CUDA_C2HS_COMPAT_H

#ifndef _Alignas
#  define _Alignas(x)   /* _Alignas elided for c2hs */
#endif
#ifndef _Static_assert
#  define _Static_assert(e, m)  /* _Static_assert elided for c2hs */
#endif
#ifndef _Noreturn
#  define _Noreturn     /* _Noreturn elided for c2hs */
#endif
#ifndef _Alignof
#  define _Alignof(x)   __alignof__(x)
#endif

#endif /* CUDA_C2HS_COMPAT_H */
