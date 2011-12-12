AC_INIT([Haskell CUDA bindings], [0.4.1.0], [tmcdonell@cse.unsw.edu.au], [cuda])
AC_CONFIG_SRCDIR([Foreign/CUDA.hs])
AC_CONFIG_FILES([cuda.buildinfo])
AC_PROG_CC

# Determine the target platform
#
AC_CANONICAL_TARGET

# Search the user's PATH for the 'nvcc' compiler. If it is found, add this
# prefix to the include and library search directories. Additionally, set nvcc
# as the C preprocessor for c2hs (only, or it won't try to link cudart)
#
AC_PATH_PROG(NVCC, nvcc)
if test "$NVCC" != ""; then
    cuda_prefix="$(dirname $(dirname $NVCC))"
    cuda_c2hsflags="--cpp=$NVCC --cppopts=-E "
    CPPFLAGS+=" -I${cuda_prefix}/include "
#    CC=${NVCC}

    case $target in
    x86_64*) LDFLAGS+=" -L${cuda_prefix}/lib64 " ;;
    *)       LDFLAGS+=" -L${cuda_prefix}/lib "   ;;
    esac
fi

# Recent versions of Mac OS X (10.6 and later) provides a C extension for
# creating lambda-like closure expressions (blocks), the syntax for which
# confuses the c2hs preprocessor. We disable this by undefining the __BLOCKS__
# macro.
#
AC_MSG_CHECKING(for Apple Blocks extension)
if test -r "/usr/include/stdlib.h"; then
    BLOCKS=`grep __BLOCKS__ /usr/include/stdlib.h`
fi
if test "$BLOCKS" != ""; then
    cuda_c2hsflags+="--cppopts=-U__BLOCKS__ "
    AC_MSG_RESULT(yes)
else
    AC_MSG_RESULT(no)
fi

# If we are running on Mac OS add the CUDA library path to the search list. This
# option allows applications to run without requiring to set [DY]LD_LIBRARY_PATH
#
case $build in
    *darwin* ) LDFLAGS+=" -Xlinker -rpath ${cuda_prefix}/lib " ;;
    * ) ;;
esac

# Make sure both the driver and runtime are found
#
longerror='
********************************************************************************

The configuration process failed to locate your CUDA installation. Ensure that
you have installed the driver and developer toolkit, available from:

  http://developer.nvidia.com/cuda-downloads

Then make sure that "nvcc" is available in your PATH, or set the appropriate
directories with --extra-include-dirs and --extra-lib-dirs.

********************************************************************************'

AC_CHECK_HEADERS([cuda.h cuda_runtime_api.h], [], [AC_MSG_ERROR(could not find CUDA headers${longerror})])
AC_SEARCH_LIBS(cuDriverGetVersion,    cuda,   [], [AC_MSG_ERROR(could not find CUDA driver library${longerror})])
AC_SEARCH_LIBS(cudaRuntimeGetVersion, cudart, [], [AC_MSG_ERROR(could not find CUDA runtime library${longerror})])

# Populate the buildinfo, with the search paths and any target specific options
#
cuda_cppflags="$CPPFLAGS"
cuda_ldflags="$LDFLAGS $LIBS"

AC_SUBST([cuda_cppflags])
AC_SUBST([cuda_ldflags])
AC_SUBST([cuda_c2hsflags])
AC_OUTPUT

