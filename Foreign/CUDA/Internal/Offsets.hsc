{-
 - Structure field offset constants.
 - Too difficult to extract using C->Haskell )=
 -}

module Foreign.CUDA.Internal.Offsets where

#include <cuda_runtime_api.h>

devNameOffset, devMaxThreadDimOffset, devMaxGridSizeOffset :: Int

devNameOffset         = #{offset struct cudaDeviceProp, name}
devMaxThreadDimOffset = #{offset struct cudaDeviceProp, maxThreadsDim}
devMaxGridSizeOffset  = #{offset struct cudaDeviceProp, maxGridSize}

