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
    Texture(..), FormatKind(..), AddressMode(..), FilterMode(..), FormatDesc(..),
    getTex, bind, bind2D, unbind
  )
  where

-- Friends
import Foreign.CUDA.Ptr
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Internal.Offsets

-- System
import Data.Int
import Foreign
import Foreign.C

#include <cuda_runtime_api.h>
{# context lib="cudart" #}

#c
typedef struct textureReference      textureReference;
typedef struct cudaChannelFormatDesc cudaChannelFormatDesc;
#endc

--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |A texture reference
--
{# pointer *textureReference as ^ foreign -> Texture nocode #}

data Texture = Texture
  {
    normalised :: Bool,         -- ^ access texture using normalised coordinates [0.0,1.0]
    filtering  :: FilterMode,
    addressing :: (AddressMode, AddressMode, AddressMode),
    format     :: FormatDesc
  }
  deriving (Eq, Show)

-- |Texture channel format kind
--
{# enum cudaChannelFormatKind as FormatKind
  { }
  with prefix="cudaChannelFormatKind" deriving (Eq, Show) #}

-- |Texture addressing mode
--
{# enum cudaTextureAddressMode as AddressMode
  { }
  with prefix="cudaAddressMode" deriving (Eq, Show) #}

-- |Texture filtering mode
--
{# enum cudaTextureFilterMode as FilterMode
  { }
  with prefix="cudaFilterMode" deriving (Eq, Show) #}


-- |A description of how memory read through the texture cache should be
-- interpreted, including the kind of data and the number of bits of each
-- component (x,y,z and w, respectively).
--
{# pointer *cudaChannelFormatDesc as ^ foreign -> FormatDesc nocode #}

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


instance Storable Texture where
  sizeOf    _ = {# sizeof textureReference #}
  alignment _ = alignment (undefined :: Ptr ())

  peek p = do
    n  <- cToBool `fmap` {# get textureReference.normalized #} p
    fm <- cToEnum `fmap` {# get textureReference.filterMode #} p
    am <- {# get textureReference.addressMode #} p
    a0 <- cToEnum `fmap` peekElemOff am 0
    a1 <- cToEnum `fmap` peekElemOff am 1
    a2 <- cToEnum `fmap` peekElemOff am 2
    cd <- peekByteOff p devTexChannelDescOffset
    return $ Texture n fm (a0,a1,a2) cd


--------------------------------------------------------------------------------
-- Texture References
--------------------------------------------------------------------------------

-- |Bind the memory area associated with the device pointer to a texture
-- reference. Any previously bound references are unbound.
--
bind :: Texture -> FormatDesc -> DevicePtr a -> Int64 -> IO ()
bind tex desc dptr bytes = nothingIfOk =<< cudaBindTexture tex dptr desc bytes

{# fun unsafe cudaBindTexture
  { alloca- `Int'
  , with_*  `Texture'
  , dptr    `DevicePtr a'
  , with_*  `FormatDesc'
  ,         `Int64'       } -> `Status' cToEnum #}
  where dptr = useDevicePtr . castDevPtr

-- |Bind the two-dimensional memory area to a texture reference. The size of the
-- area is constrained by (width,height) in texel units, and the row pitch in
-- bytes. Any previously bound references are unbound.
--
bind2D :: Texture -> FormatDesc -> DevicePtr a -> (Int,Int) -> Int64 -> IO ()
bind2D tex desc dptr (width,height) bytes =
  nothingIfOk =<< cudaBindTexture2D tex dptr desc width height bytes

{# fun unsafe cudaBindTexture2D
  { alloca- `Int'
  , with_*  `Texture'
  , dptr    `DevicePtr a'
  , with_*  `FormatDesc'
  ,         `Int'
  ,         `Int'
  ,         `Int64'       } -> `Status' cToEnum #}
  where dptr = useDevicePtr . castDevPtr


-- |Returns the texture reference associated with the given symbol
--
getTex :: String -> IO (Texture)
getTex name =
  allocaBytes (sizeOf (undefined :: Ptr ())) $ \p ->
    peek =<< resultIfOk =<< cudaGetTextureReference p name

{# fun unsafe cudaGetTextureReference
  { with_*       `Ptr Texture' peek*
  , withCString* `String'            } -> `Status' cToEnum #}

-- |Unbind a texture reference
--
unbind :: Texture -> IO ()
unbind tex = nothingIfOk =<< cudaUnbindTexture tex

{# fun unsafe cudaUnbindTexture
  { with_* `Texture' } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

with_ :: Storable a => a -> (Ptr a -> IO b) -> IO b
with_ = with

