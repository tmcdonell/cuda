AC_INIT([Haskell CUDA bindings], [0.2.1], [tmcdonell@cse.unsw.edu.au], [cuda])
AC_CONFIG_SRCDIR([Foreign/CUDA.hs])
AC_CONFIG_FILES([cuda.buildinfo])
AC_PROG_CXX

# Search the user's PATH for the 'nvcc' compiler. If it is found, add this
# prefix to the include and library search directories.
#
AC_PATH_PROG(NVCC, nvcc)
if test "$NVCC" != ""; then
    cuda_prefix="$(dirname $(dirname $NVCC))"
    CPPFLAGS+="-I${cuda_prefix}/include"
    LDFLAGS+="-L${cuda_prefix}/lib"
fi

# Target specific options
# - Snow Leopard (Mac OS X 10.6) provides a C extension for creating
#   lambda-like closure expressions (blocks), the syntax for which confuses the
#   c2hs preprocessor. We disable this by undefining the __BLOCKS__ macro.
#
AC_MSG_CHECKING(for Apple Blocks extension)
if test -r "/System/Library/CoreServices/SystemVersion.plist" \
        -a "`grep '<string>10.6' /System/Library/CoreServices/SystemVersion.plist`" != ""; then
    cuda_c2hsflags+="-C-U__BLOCKS__"
    AC_MSG_RESULT(yes)
else
    AC_MSG_RESULT(no)
fi

# Make sure both the driver and runtime are found
#
AC_CHECK_HEADERS([cuda.h cuda_runtime_api.h], [], [AC_MSG_ERROR(could not find CUDA headers)])
AC_SEARCH_LIBS(cuDriverGetVersion,    cuda,   [], [AC_MSG_ERROR(could not find CUDA driver library)])
AC_SEARCH_LIBS(cudaRuntimeGetVersion, cudart, [], [AC_MSG_ERROR(could not find CUDA runtime library)])

# Populate the buildinfo, with the search paths and any target specific options
#
cuda_cppflags="$CPPFLAGS"
cuda_ldflags="$LDFLAGS"

AC_SUBST([cuda_cppflags])
AC_SUBST([cuda_ldflags])
AC_SUBST([cuda_c2hsflags])
AC_OUTPUT

