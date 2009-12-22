// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5632 $
// $Date: 2009-07-01 14:36:01 +1000 (Wed, 01 Jul 2009) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

#ifndef __TYPE_VECTOR_H__
#define __TYPE_VECTOR_H__

/** @brief Utility template struct for generating small vector types from scalar types
  *
  * Given a base scalar type (\c int, \c float, etc.) and a vector length (1 through 4) as
  * template parameters, this struct defines a vector type (\c float3, \c int4, etc.) of the
  * specified length and base type.  For example:
  * \code
  * template <class T>
  * __device__ void myKernel(T *data)
  * {
  *     typeToVector<T,4>::Result myVec4;             // create a vec4 of type T
  *     myVec4 = (typeToVector<T,4>::Result*)data[0]; // load first element of data as a vec4
  * }
  * \endcode
  *
  * This functionality is implemented using template specialization.  Currently specializations
  * for int, float, and unsigned int vectors of lengths 2-4 are defined.  Note that this results
  * in types being generated at compile time -- there is no runtime cost.  typeToVector is used by
  * the optimized scan \c __device__ functions in scan_cta.cu.
  */
template <typename T, int N>
struct typeToVector
{
    typedef T Result;
};

#define TYPE_VECTOR(type, name)                                                 \
    template <> struct typeToVector<type, 2> { typedef name##2 Result; };       \
    template <> struct typeToVector<type, 3> { typedef name##3 Result; };       \
    template <> struct typeToVector<type, 4> { typedef name##4 Result; }

TYPE_VECTOR(int,         int);
TYPE_VECTOR(float,       float);
TYPE_VECTOR(unsigned int, uint);


#undef TYPE_VECTOR
#endif

