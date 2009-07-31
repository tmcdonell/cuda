#ifndef __CSTUBS_H
#define __CSTUBS_H

#ifdef __cplusplus
extern "C" {
#endif

void cuda_plus_scan(float *out, float *in, int n);

#ifdef __cplusplus
}
#endif
#endif
