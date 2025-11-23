/* -----------------------------------------------------------------------------
 *
 * Module    : Operator
 * Copyright : (c) 2009 Trevor L. McDonell
 * License   : BSD
 *
 * A template class for unary/binary kernel operations
 *
 * ---------------------------------------------------------------------------*/


#ifndef __OPERATOR_H__
#define __OPERATOR_H__

#include <float.h>
#include <limits.h>


/*
 * Template class for an operation that can be mapped over an array.
 */
template <typename Ta, typename Tb=Ta>
class Functor
{
public:
    /*
     * Apply the operation to the given operand.
     */
    static __device__ Tb apply(const Ta &x);
};

template <typename Ta, typename Tb>
class fromIntegral : Functor<Ta, Tb>
{
public:
    static __device__ Tb apply(const Ta &x) { return (Tb) x; }
};


/*
 * Template class for binary operators. Certain algorithms may require the
 * operator to be associative (that is, Ta == Tb), such as parallel scan and
 * reduction.
 *
 * As this is template code, it should compile down to something efficient...
 */
template <typename Ta, typename Tb=Ta, typename Tc=Ta>
class BinaryOp
{
public:
    /*
     * Apply the operation to the given operands.
     */
    static __device__ Tc apply(const Ta &a, const Tb &b);

    /*
     * Return an identity element for the type Tc.
     *
     * This may have special meaning for a given implementation, for example a
     * `max' operation over integers may want to return INT_MIN.
     */
    static __device__ Tc identity();
};

/*
 * Return the minimum or maximum value of a type
 */
template <typename T> inline __device__ T getMin();
template <typename T> inline __device__ T getMax();

#define SPEC_MINMAX(type,vmin,vmax)                                            \
    template <> inline __device__ type getMin() { return vmin; };              \
    template <> inline __device__ type getMax() { return vmax; }               \

SPEC_MINMAX(float,        -FLT_MAX,  FLT_MAX);
SPEC_MINMAX(int,           INT_MIN,  INT_MAX);
SPEC_MINMAX(char,          CHAR_MIN, CHAR_MAX);
SPEC_MINMAX(unsigned int,  0,        UINT_MAX);
SPEC_MINMAX(unsigned char, 0,        UCHAR_MAX);


/*
 * Basic binary arithmetic operations. We take advantage of automatic type
 * promotion to keep the parameters general.
 */
#define BASIC_OP(name,expr,id)                                                 \
    template <typename Ta, typename Tb=Ta, typename Tc=Ta>                     \
    class name : BinaryOp<Ta, Tb, Tc>                                          \
    {                                                                          \
    public:                                                                    \
        static __device__ Tc apply(const Ta &a, const Tb &b) { return expr; }  \
        static __device__ Tc identity() { return id; }                         \
    }

#define LOGICAL_OP(name,expr,id)                                               \
    template <typename Ta, typename Tb=Ta>                                     \
    class name : BinaryOp<Ta, Tb, bool>                                        \
    {                                                                          \
    public:                                                                    \
        static __device__ bool apply(const Ta &a, const Tb &b) { return expr; }\
        static __device__ bool identity() { return id; }                       \
    }

BASIC_OP(Plus,  a + b,    0);
BASIC_OP(Times, a * b,    1);
BASIC_OP(Min,   min(a,b), getMax<Ta>());
BASIC_OP(Max,   max(a,b), getMin<Ta>());

LOGICAL_OP(Eq,  a == b,   false);

#undef SPEC_MINMAX
#undef BASIC_OP
#endif

