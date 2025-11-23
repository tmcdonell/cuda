/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

/* Thread block size */
#define BLOCK_SIZE 16

/* Matrix dimensions
 * (chosen as multiples of the thread block size for simplicity)
 */
#define WA (3 * BLOCK_SIZE) /* Matrix A width  */
#define HA (5 * BLOCK_SIZE) /* Matrix A height */
#define WB (8 * BLOCK_SIZE) /* Matrix B width  */
#define HB WA  /* Matrix B height */
#define WC WB  /* Matrix C width  */
#define HC HA  /* Matrix C height */

#endif /* _MATRIXMUL_H_ */

