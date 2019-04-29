Using the CUDA package on Windows
=================================

The CUDA package works on Windows and is actively maintained. If you encounter
any other issues, please report them.

Note that if you build your applications for the Windows 64-bit architecture,
you'll need to update your `ld.exe` as described below.


Windows 64-bit
--------------

There is a known issue with the version of `ld.exe` that ships with the 64-bit
versions of (at least) GHC-7.8.4 and GHC-7.10.2. The version of `ld.exe` that
ships with these GHC distributions does not properly link against MS-style
dynamic libraries (such as those that ship with the CUDA toolkit), causing the
application to crash at runtime once those library routines are called. The
configure step will fail if it detects an old version of `ld.exe` (< 2.25.1),
which are known to be broken.

If you are using the 64-bit GHC distributions mentioned above, you will need to
apply the following steps. This bug does not affect 32-bit GHC distributions.
The bug has been fixed in MinGW binutils `ld.exe` >= 2.25.1, so it is expected
that newer releases of GHC will not have this issue.

The problem is fixed by replacing the linker binary `ld.exe` with the newer
(patched) version, available as part of the MSys2 binutils package here:

> <http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-binutils-2.25.1-1-any.pkg.tar.xz>

The updated `ld.exe` binary must replace the version at the path:

> `GHC_PATH\mingw\x86_64-w64-mingw32\bin\`

Note that there is another copy of `ld.exe` located at `GHC_PATH\mingw\bin\`,
but this version does not seem to be used, so replacing it as well is not
necessary. It is not sufficient to replace whatever version of `ld.exe` appears
first in your `PATH`.

Please note that having another MinGW installation in `PATH` before the one
shipped with GHC may break things, particularly if you mix 32/64-bit
distributions of MinGW and GHC.

For further discussion of the bug, see:

  * [CUDA package issue][cuda31]
  * [GHC issue][ghc10885]
  * [binutils issue][binutils16598]

 [cuda31]:              https://github.com/tmcdonell/cuda/issues/31
 [ghc10885]:            https://ghc.haskell.org/trac/ghc/ticket/10885
 [binutils16598]:       https://sourceware.org/bugzilla/show_bug.cgi?id=16598

