{-# LANGUAGE ForeignFunctionInterface, EmptyDataDecls #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Texture
-- Copyright : (c) [2009..2010] Trevor L. McDonell
-- License   : BSD
--
-- Texture references
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Texture
  (
    Texture, FormatKind(..), FormatDesc(..),
    getTex, setPtr, unbind
  )
  where

-- Friends
import Foreign.CUDA.Ptr
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Data.Int
import Foreign
import Foreign.C

#include "cbits/stubs.h"
#include <cuda_runtime_api.h>
{# context lib="cudart" #}

#c
typedef struct cudaChannelFormatDesc cudaChannelFormatDesc;
#endc

--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |A texture reference
--
{# pointer *textureReference as Texture foreign newtype #}
withTexture :: Texture -> (Ptr Texture -> IO b) -> IO b

-- |Texture channel format kind
--
{# enum cudaChannelFormatKind as FormatKind
  { }
  with prefix="cudaChannelFormatKind" deriving (Eq, Show) #}


{# pointer *cudaChannelFormatDesc as ^ foreign -> FormatDesc nocode #}

-- |A description of how memory read through the texture cache should be
-- interpreted, including the kind of data and the number of bits of each
-- component (x,y,z and w, respectively).
--
data FormatDesc = FormatDesc
  {
    depth :: (Int,Int,Int,Int),
    kind  :: FormatKind
  }
  deriving (Eq, Show)

instance Storable FormatDesc where
  sizeOf    _ = {# sizeof cudaChannelFormatDesc #}
  alignment _ = alignment (undefined :: Ptr ())

  peek p = do
    dx <- cIntConv `fmap` {# get cudaChannelFormatDesc.x #} p
    dy <- cIntConv `fmap` {# get cudaChannelFormatDesc.y #} p
    dz <- cIntConv `fmap` {# get cudaChannelFormatDesc.z #} p
    dw <- cIntConv `fmap` {# get cudaChannelFormatDesc.w #} p
    df <- cToEnum  `fmap` {# get cudaChannelFormatDesc.f #} p
    return $ FormatDesc (dx,dy,dz,dw) df

  poke p (FormatDesc (x,y,z,w) k) = do
    {# set cudaChannelFormatDesc.x #} p (cIntConv x)
    {# set cudaChannelFormatDesc.y #} p (cIntConv y)
    {# set cudaChannelFormatDesc.z #} p (cIntConv z)
    {# set cudaChannelFormatDesc.w #} p (cIntConv w)
    {# set cudaChannelFormatDesc.f #} p (cFromEnum k)


--------------------------------------------------------------------------------
-- Texture References
--------------------------------------------------------------------------------

-- |Bind the memory area associated with the device pointer to a texture
-- reference. Any previously bound references are unbound.
--
setPtr :: Texture -> FormatDesc -> DevicePtr a -> Int64 -> IO ()
setPtr tex desc dptr bytes = nothingIfOk =<< cudaBindTexture tex dptr desc bytes

{# fun unsafe cudaBindTexture
  { alloca-      `Int'
  , withTexture* `Texture'
  , dptr         `DevicePtr a'
  , with'*       `FormatDesc'
  ,              `Int64'       } -> `Status' cToEnum #}
  where
    with' = with
    dptr  = useDevicePtr . castDevPtr


-- |Returns the texture reference associated with the given symbol
--
getTex :: String -> IO (Texture)
getTex name =
  allocaBytes (sizeOf (undefined :: Ptr ())) $ \p -> do
    tex <- resultIfOk =<< cudaGetTextureReference p name
    Texture `fmap` newForeignPtr_ tex

{# fun unsafe cudaGetTextureReference
  { with'*       `Ptr Texture' peek*
  , withCString* `String'            } -> `Status' cToEnum #}
  where with' = with

-- |Unbind a texture reference
--
unbind :: Texture -> IO ()
unbind tex = nothingIfOk =<< cudaUnbindTexture tex

{# fun unsafe cudaUnbindTexture
  { withTexture* `Texture' } -> `Status' cToEnum #}

