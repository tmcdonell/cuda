# Change Log

Notable changes to the project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/).

## [0.9.0.3] - 2018-03-12
### Fixed
  * Build fix for Cabal-2.2 (ghc-8.4)

## [0.9.0.2] - 2018-03-07
### Fixed
  * Build fix for Nix ([#53])

## [0.9.0.1] - 2018-02-16
### Fixed
  * Build fix for macOS High Sierra (10.13)

## [0.9.0.0] - 2017-11-15
### Fixed
  * Build fixes for CUDA-9

### Added
  * `Peer.getAttribute`
  * `Exec.launchKernelCooperative`

### Changed
  * Changed type of `Stream.wait` and `Stream.write` to support 64-bit values

## [0.8.0.1] - 2017-10-24
### Fixed
  * Escape backslashes used in -D flags on Windows ([#50])

## [0.8.0.0] - 2017-08-24
### Changed
  * Tested with CUDA toolkit 8.0

### Added
  * Add operations for unified addressing in the device API
  * Add `write` and `wait` operations for streams in the device API
  * (internals) The paths this module was configured against are exposed by the
    module `Foreign.CUDA.Paths`.

## [0.7.5.3] - 2017-03-23
### Fixed
  * Bug fix in occupancy calculator

## [0.7.5.2] - 2017-01-06
### Fixed
  * Build fails with library profiling ([#43])
  * On Windows, the Cabal installer is looking in the wrong place ([#45])
  * Windows install fix ([#47])

## [0.7.5.1] - 2016-10-21
### Fixed
  * Re-enable support for Cabal-1.22
  * Unknown CUDA device compute capability 6.1 ([#40])
  * Compilation fails for CUDA-8 [was: ghc 7.10.3 fail to install] ([#44])

## [0.7.5.0] - 2016-10-07
### Changed
  * Tested with CUDA toolkit 7.5

### Added
  * Add functions from CUDA-7.5
  * Add profiler control functions
  * Add function `mallocHostForeignPtr`

## [0.7.0.0] - 2015-11-30
### Changed
  * Add support for operations from CUDA-7.0
  * Add support for online linking
  * Add support for inter-process communication
  * Bug fixes, extra documentation, improve library coverage.
  * Mac OS X no longer requires the DYLD_LIBRARY_PATH environment variable in
    order to compile or run programs that use this package.

## [0.6.7.0] - 2015-09-12
### Added
  * Add support for building on Windows (thanks to @mwu-tow)

## [0.6.6.2] - 2015-04-04
### Fixed
  * Build fix

## [0.6.6.1] - 2015-04-04 [YANKED]
### Fixed
  * Build fixes for ghc-7.6 and ghc-7.10

## [0.6.6.0] - 2015-03-10
### Added
  * Add compute-capability data for 3.7, 5.2 devices.

### Changed
  * Combine the definition of the 'Event' and 'Stream' data types. As of
    CUDA-3.1 these data structures are equivalent, and can be safely shared
    between runtime and driver API calls and libraries.

  * Mark FFI imports of potentially long-running API functions as safe. This
    allows them to be safely called from Haskell threads without blocking the
    entire HEC.

### Removed
  * Drop support for CUDA 3.0 and older.

## [0.6.5.1] - 2014-12-02
### Fixed
  * Build fix for Mac OS X 10.10 (Yosemite)

## [0.6.5.0] - 2014-09-03
### Changed
  * Tested with CUDA toolkit 6.5

### Added
  * Add functions from CUDA-6.5


[0.9.0.3]:    https://github.com/tmcdonell/cuda/compare/0.9.0.2...0.9.0.3
[0.9.0.2]:    https://github.com/tmcdonell/cuda/compare/0.9.0.1...0.9.0.2
[0.9.0.1]:    https://github.com/tmcdonell/cuda/compare/0.9.0.0...0.9.0.1
[0.9.0.0]:    https://github.com/tmcdonell/cuda/compare/0.8.0.1...0.9.0.0
[0.8.0.1]:    https://github.com/tmcdonell/cuda/compare/0.8.0.0...0.8.0.1
[0.8.0.0]:    https://github.com/tmcdonell/cuda/compare/0.7.5.3...0.8.0.0
[0.7.5.3]:    https://github.com/tmcdonell/cuda/compare/0.7.5.2...0.7.5.3
[0.7.5.2]:    https://github.com/tmcdonell/cuda/compare/0.7.5.1...0.7.5.2
[0.7.5.1]:    https://github.com/tmcdonell/cuda/compare/0.7.5.0...0.7.5.1
[0.7.5.0]:    https://github.com/tmcdonell/cuda/compare/0.7.0.0...0.7.5.0
[0.7.0.0]:    https://github.com/tmcdonell/cuda/compare/0.6.7.0...0.7.0.0
[0.6.7.0]:    https://github.com/tmcdonell/cuda/compare/0.6.6.2...0.6.7.0
[0.6.6.2]:    https://github.com/tmcdonell/cuda/compare/0.6.6.1...0.6.6.2
[0.6.6.1]:    https://github.com/tmcdonell/cuda/compare/0.6.6.0...0.6.6.1
[0.6.6.0]:    https://github.com/tmcdonell/cuda/compare/0.6.5.1...0.6.6.0
[0.6.5.1]:    https://github.com/tmcdonell/cuda/compare/0.6.5.0...0.6.5.1
[0.6.5.0]:    https://github.com/tmcdonell/cuda/compare/0.6.0.1...0.6.5.0

[#40]:        https://github.com/tmcdonell/cuda/issues/40
[#43]:        https://github.com/tmcdonell/cuda/issues/43
[#44]:        https://github.com/tmcdonell/cuda/issues/44
[#45]:        https://github.com/tmcdonell/cuda/issues/45
[#47]:        https://github.com/tmcdonell/cuda/pull/47
[#50]:        https://github.com/tmcdonell/cuda/pull/50
[#53]:        https://github.com/tmcdonell/cuda/pull/53

