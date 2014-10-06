{-# LANGUAGE ForeignFunctionInterface #-}
{-
 - Structure field offset constants.
 - Too difficult to extract using C->Haskell )=
 -}

module Foreign.CUDA.Internal.Offsets where


--------------------------------------------------------------------------------
-- Runtime API
--------------------------------------------------------------------------------

#include "cbits/stubs.h"

devNameOffset, devMaxThreadDimOffset, devMaxGridSizeOffset :: Int

devNameOffset         = #{offset struct cudaDeviceProp, name}
devMaxThreadDimOffset = #{offset struct cudaDeviceProp, maxThreadsDim}
devMaxGridSizeOffset  = #{offset struct cudaDeviceProp, maxGridSize}

#if CUDART_VERSION >= 3000
devMaxTexture2DOffset, devMaxTexture3DOffset :: Int
devMaxTexture2DOffset = #{offset struct cudaDeviceProp, maxTexture2D}
devMaxTexture3DOffset = #{offset struct cudaDeviceProp, maxTexture3D}
#endif

texRefAddressModeOffset, texRefChannelDescOffset :: Int
texRefAddressModeOffset = #{offset struct textureReference, addressMode}
texRefChannelDescOffset = #{offset struct textureReference, channelDesc}

--------------------------------------------------------------------------------
-- Driver API
--------------------------------------------------------------------------------

#include <cuda.h>

devMaxThreadDimOffset', devMaxGridSizeOffset' :: Int

devMaxThreadDimOffset' = #{offset struct CUdevprop_st, maxThreadsDim}
devMaxGridSizeOffset'  = #{offset struct CUdevprop_st, maxGridSize}

