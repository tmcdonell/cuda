AC_INIT([Haskell CUDA bindings], [0.5.0.0], [tmcdonell@cse.unsw.edu.au], [cuda])
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
AC_ARG_WITH([compiler], [Haskell compiler], [GHC=$withval],  [AC_PATH_PROG(GHC, ghc)])
AC_ARG_WITH([nvcc],     [CUDA compiler],    [NVCC=$withval], [AC_PATH_PROG(NVCC, nvcc)])
AC_ARG_WITH([gcc],      [C compiler],       [CC=$withval])

# If NVCC is detected in the path, set the location of the toolkit relative to
# that, otherwise choose the default
#
if test "$NVCC" != ""; then
    cuda_prefix="$(dirname "$(dirname "$NVCC")")"
    cuda_c2hsflags="--cpp="$NVCC" --cppopts=-E "
else
    cuda_prefix="/usr/local/cuda"
fi

# Location of the toolkit binaries and libraries
#
cuda_inc_path="${cuda_prefix}/include"
CPPFLAGS+=" -I${cuda_inc_path} "

case $target_os in
  darwin*)
    cuda_lib_path="${cuda_prefix}/lib"
    LDFLAGS+=" -Xlinker -rpath ${cuda_lib_path} -F/Library/Frameworks -framework CUDA"
    #CPPFLAGS+=" -arch ${target_cpu} "
    ;;
  *)
    case $target_cpu in
      x86_64*)
        cuda_lib_path="${cuda_prefix}/lib64"
        #CPPFLAGS+=" -m64 "
        ;;
      *)
        cuda_lib_path="${cuda_prefix}/lib"
        #CPPFLAGS+=" -m32 "
        ;;
    esac
    ;;
esac
LDFLAGS+=" -L${cuda_lib_path} "


# The CUDA headers need to be interpreted relative to the architecture GHC is
# compiled for, which might be different from the architecture of the host
# machine (eg. 32-bit GHC on a 64-bit CPU).
#
AC_MSG_CHECKING(ghc architecture)
target_ghc_arch=`${GHC} -ignore-dot-ghci -e "putStrLn System.Info.arch"`
AC_MSG_RESULT(${target_ghc_arch})

case $target_ghc_arch in
  x86_64)       cuda_c2hsflags+="--cppopts=-m64 " ;;
  i386)         cuda_c2hsflags+="--cppopts=-m32 " ;;
  *)            ;;
esac

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

# Make sure both the driver and runtime are found
#
longerror='
********************************************************************************

The configuration process failed to locate your CUDA installation. Ensure that
you have installed both the developer driver and toolkit, available from:

  http://developer.nvidia.com/cuda-downloads

and make sure that "nvcc" is available in your PATH. Check the above output log
and run the command directly to ensure it can be located.

If you have a non-standard installation, you can add additional search paths
using --extra-include-dirs and --extra-lib-dirs. Note that 64-bit Linux flavours
often require both "lib64" and "lib" library paths, in that order.

********************************************************************************'

AC_CHECK_HEADERS([cuda.h cuda_runtime_api.h], [], [AC_MSG_ERROR(could not find CUDA headers${longerror})])
AC_SEARCH_LIBS(cuDriverGetVersion,    cuda,   [], [AC_MSG_ERROR(could not find CUDA driver library${longerror})])
AC_SEARCH_LIBS(cudaRuntimeGetVersion, cudart, [], [AC_MSG_ERROR(could not find CUDA runtime library${longerror})])

# Populate the buildinfo, with the search paths and any target specific options
#
cuda_ccflags="$CPPFLAGS "
cuda_ldflags="$LDFLAGS $LIBS "

for x in $cuda_ccflags; do
    cuda_ghcflags+="-optc${x} "
done
for x in $cuda_ldflags; do
    cuda_ghcflags+="-optl${x} "
done

AC_SUBST([cuda_ghcflags])
AC_SUBST([cuda_ccflags])
AC_SUBST([cuda_ldflags])
AC_SUBST([cuda_c2hsflags])
AC_OUTPUT

