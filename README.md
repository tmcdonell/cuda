Haskell FFI Bindings to CUDA
============================

[![CI-Linux](https://github.com/tmcdonell/cuda/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/tmcdonell/cuda/actions/workflows/ci-linux.yml)
[![CI-Windows](https://github.com/tmcdonell/cuda/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/tmcdonell/cuda/actions/workflows/ci-windows.yml)
[![Stackage LTS](https://stackage.org/package/cuda/badge/lts)](https://stackage.org/lts/package/cuda)
[![Stackage Nightly](https://stackage.org/package/cuda/badge/nightly)](https://stackage.org/nightly/package/cuda)
[![Hackage](https://img.shields.io/hackage/v/cuda.svg)](https://hackage.haskell.org/package/cuda)

The CUDA library provides a direct, general purpose C-like SPMD programming
model for NVIDIA graphics cards (G8x series onwards). This is a collection of
bindings to allow you to call and control, although not write, such functions
from Haskell-land. You will need to install the CUDA driver and developer
toolkit.

  <http://developer.nvidia.com/object/cuda.html>

The configure step will look for your CUDA installation in the standard places,
and if the `nvcc` compiler is found in your `PATH`, relative to that.

For important information on installing on Windows, see:

  <https://github.com/tmcdonell/cuda/blob/master/WINDOWS.md>


## Missing functionality

_This library is currently in **maintenance mode**. While we plan to release
updates to keep the existing interface working with newer CUDA versions (as
long as the underlying APIs remain available), no binding of new features is
planned at the moment. Get in touch if you want to contribute._

Here is an incomplete historical list of missing bindings. Pull requests welcome!

### CUDA-9

- cuLaunchCooperativeKernelMultiDevice

### CUDA-10.0

- cuDeviceGetLuid (windows only?)
- cuLaunchHostFunc
- cuGraphHostNode[Get/Set]Params
- cuGraphKernelNode[Get/Set]Params
- cuGraphMemcpyNode[Get/Set]Params
- cuGraphMemsetNode[Get/Set]Params

### CUDA-10.2

- cuDeviceGetNvSciSyncAttributes
- cuMemAddressFree
- cuMemAddressReserve
- cuMemCreate
- cuMemExportToShareableHandle
- cuMemGetAccess
- cuMemGetAllocationGranularity
- cuMemGetAllocationPrepertiesFromHandle
- cuMemImportFromShareableHandle
- cuMemMap
- cuMemRelease
- cuMemSetAccess
- cuMemUnmap
- cuGraphExecHostNodeSetParams
- cuGraphExecMemcpyNodeSetParams
- cuGraphExecMemsetNodeSetParams
- cuGraphExecUpdate

### CUDA-11.0

- cuCtxResetPersistentingL2Cache
- cuMemRetainAllocationHandle
- cuStreamCopyAttributes
- cuStreamGetAttribute
- cuStreamSetAttribute
- cuGraphKernelNodeCopyAttributes
- cuGraphKernelNodeGetAttribute
- cuGraphKernelNodeSetAttribute
- cuOccupancyAvailableDynamicSMemPerBlock

### CUDA-11.1

- cuDeviceGetTexture1DLinearMaxWidth
- cuArrayGetSparseProperties
- cuMipmappedArrayGetSparseProperties
- cuMemMapArrayAsync
- cuEventRecordWithFlags
- cuGraphAddEventRecordNode
- cuGraphAddEventWaitNode
- cuGraphEventRecordNodeGetEvent
- cuGraphEventRecordNodeSetEvent
- cuGraphEventWaitNodeGetEvent
- cuGraphEventWaitNodeSetEvent
- cuGraphExecChildGraphNodeSetParams
- cuGraphExecEventRecordNodeSetEvent
- cuGraphExecEventWaitNodeSetEvent
- cuGraphUpload

### CUDA-11.2

- cuDeviceGetDefaultMemPool
- cuDeviceGetMemPool
- cuDeviceSetMemPool
- cuArrayGetPlane
- cuMemAllocAsync
- cuMemAllocFromPoolAsync
- cuMemFreeAsync
- cuMemPoolCreate
- cuMemPoolDestroy
- cuMemPoolExportPointer
- cuMemPoolExportToShareableHandle
- cuMemPoolGetAccess
- cuMemPoolGetAttribute
- cuMemPoolImportFromShareableHandle
- cuMemPoolImportPointer
- cuMemPoolSetAccess
- cuMemPoolSetAttribute
- cuMemPoolTrimTo
- cuGraphAddExternalSemaphoresSignalNode
- cuGraphAddExternalSemaphoresWaitNode
- cuGraphExecExternalSemaphoresSignalNodeSetParams
- cuGraphExecExternalSemaphoresWaitNodeSetParams
- cuGraphExternalSemaphoresSignalNodeGetParams
- cuGraphExternalSemaphoresSignalNodeSetParams
- cuGraphExternalSemaphoresWaitNodeGetParams
- cuGraphExternalSemaphoresWaitNodeSetParams

### CUDA-11.3

- cuStreamGetCaptureInfo_v2
- cuFuncGetModule
- cuGraphDebugDotPrint
- cuGraphReleaseUserObject
- cuGraphRetainUserObject
- cuUserObjectCreate
- cuUserObjectRelease
- cuUserObjectRetain
- cuGetProcAddress

### CUDA-11.4

- cuDeviceGetUuid_v2
- cuCtxCreate_v3
- cuCtxGetExecAffinity
- cuDeviceGetGraphMemAttribute
- cuDeviceGraphMemTrim
- cuDeviceSetGraphMemAttribute
- cuGraphAddMemAllocNode
- cuGraphAddMemFreeNode
- cuGraphInstantiateWithFlags
- cuGraphMemAllocNodeGetParams
- cuGraphMemFreeNodeGetParams

### CUDA-12

A lot. PRs welcome.


# Old compatibility notes

The setup script for this package requires at least Cabal-1.24. If you run into trouble with this:

* Cabal users: ensure you are using a new `cabal` executable and have run `cabal update` anywhere in the last few years. If you have previously run `cabal install` on libraries and have a broken environment as a result, remove `~/.ghc/<platfom>/environments/default`.
* Stack users: one may attempt @stack setup --upgrade-cabal@.

Due to an interaction between GHC-8 and unified virtual address spaces in
CUDA, this package does not currently work with GHCi on ghc-8.0.1 (compiled
programs should work). See the following for more details:

* <https://github.com/tmcdonell/cuda/issues/39>
* <https://ghc.haskell.org/trac/ghc/ticket/12573>

The bug should be fixed in ghc-8.0.2 and beyond.
