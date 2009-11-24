{-# LANGUAGE ForeignFunctionInterface #-}
{-
 - Structure field offset constants.
 - Too difficult to extract using C->Haskell )=
 -}

module Foreign.CUDA.Internal.Offsets where


--------------------------------------------------------------------------------
-- Runtime API
--------------------------------------------------------------------------------

#include <cuda_runtime_api.h>

devNameOffset, devMaxThreadDimOffset, devMaxGridSizeOffset :: Int

devNameOffset         = #{offset struct cudaDeviceProp, name}
devMaxThreadDimOffset = #{offset struct cudaDeviceProp, maxThreadsDim}
devMaxGridSizeOffset  = #{offset struct cudaDeviceProp, maxGridSize}


--------------------------------------------------------------------------------
-- Driver API
--------------------------------------------------------------------------------

#include <cuda.h>

devMaxThreadDimOffset', devMaxGridSizeOffset' :: Int

devMaxThreadDimOffset' = #{offset struct CUdevprop_st, maxThreadsDim}
devMaxGridSizeOffset'  = #{offset struct CUdevprop_st, maxGridSize}

