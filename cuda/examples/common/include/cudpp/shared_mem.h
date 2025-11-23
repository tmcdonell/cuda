// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5633 $
// $Date: 2009-07-01 15:02:51 +1000 (Wed, 01 Jul 2009) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * sharedmem.h
 *
 * @brief Shared memory declaration struct for templatized types.
 *
 * Because dynamically sized shared memory arrays are declared "extern" in CUDA,
 * we can't templatize their types directly.  To get around this, we declare a
 * simple wrapper struct that will declare the extern array with a different
 * name depending on the type.  This avoids linker errors about multiple
 * definitions.
 *
 * To use dynamically allocated shared memory in a templatized __global__ or
 * __device__ function, just replace code like this:
 *
 * <pre>
 *  template<class T>
 *  __global__ void
 *  foo( T* d_out, T* d_in)
 *  {
 *      // Shared mem size is determined by the host app at run time
 *      extern __shared__  T sdata[];
 *      ...
 *      doStuff(sdata);
 *      ...
 *  }
 * </pre>
 *
 *  With this
 * <pre>
 *  template<class T>
 *  __global__ void
 *  foo( T* d_out, T* d_in)
 *  {
 *      // Shared mem size is determined by the host app at run time
 *      SharedMemory<T> smem;
 *      T* sdata = smem.getPointer();
 *      ...
 *      doStuff(sdata);
 *      ...
 *  }
 * </pre>
 */

#ifndef __SHARED_MEM_H__
#define __SHARED_MEM_H__


/** @brief Wrapper class for templatized dynamic shared memory arrays.
  *
  * This struct uses template specialization on the type \a T to declare
  * a differently named dynamic shared memory array for each type
  * (\code extern __shared__ T s_type[] \endcode).
  *
  * Currently there are specializations for the following types:
  * \c int, \c uint, \c char, \c uchar, \c short, \c ushort, \c long,
  * \c unsigned long, \c bool, \c float, and \c double. One can also specialize it
  * for user defined types.
  */
template <typename T>
struct SharedMemory
{
    /** Return a pointer to the runtime-sized shared memory array. **/
    __device__ T* getPointer()
    {
        extern __device__ void Error_UnsupportedType(); // Ensure that we won't compile any un-specialized types
        Error_UnsupportedType();
        return (T*)0;
    }
    // TODO: Use operator overloading to make this class look like a regular array
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

#define SPEC_SHAREDMEM(T, name)                                                         \
    template <> struct SharedMemory <T>                                                 \
    {                                                                                   \
        __device__ T* getPointer() { extern __shared__ T s_##name[]; return s_##name; } \
    }

SPEC_SHAREDMEM(int,    int);
SPEC_SHAREDMEM(char,   char);
SPEC_SHAREDMEM(long,   long);
SPEC_SHAREDMEM(short,  short);
SPEC_SHAREDMEM(bool,   bool);
SPEC_SHAREDMEM(float,  float);
SPEC_SHAREDMEM(double, double);

SPEC_SHAREDMEM(unsigned int,   uint);
SPEC_SHAREDMEM(unsigned char,  uchar);
SPEC_SHAREDMEM(unsigned long,  ulong);
SPEC_SHAREDMEM(unsigned short, ushort);

SPEC_SHAREDMEM(uchar4, uchar4);

#undef SPEC_SHAREDMEM
#endif // __SHARED_MEM_H__

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
