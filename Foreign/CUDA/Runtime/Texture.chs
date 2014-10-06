{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Texture
-- Copyright : (c) [2009..2012] Trevor L. McDonell
-- License   : BSD
--
-- Texture references
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Texture (

  -- * Texture Reference Management
  Texture(..), FormatKind(..), AddressMode(..), FilterMode(..), FormatDesc(..),
  bind, bind2D

) where

-- Friends
import Foreign.CUDA.Ptr
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Internal.Offsets

-- System
import Data.Int
import Foreign
import Foreign.C

#include "cbits/stubs.h"
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
{# pointer *textureReference as ^ -> Texture #}

data Texture = Texture
  {
    normalised :: !Bool,                -- ^ access texture using normalised coordinates [0.0,1.0)
    filtering  :: !FilterMode,
    addressing :: !(AddressMode, AddressMode, AddressMode),
    format     :: !FormatDesc
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
    depth :: !(Int,Int,Int,Int),
    kind  :: !FormatKind
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
    norm    <- cToBool     `fmap` {# get textureReference.normalized #} p
    fmt     <- cToEnum     `fmap` {# get textureReference.filterMode #} p
    [x,y,z] <- map cToEnum `fmap` (peekArray 3 (p `plusPtr` texRefAddressModeOffset :: Ptr CInt))
    dsc     <- peekByteOff p texRefChannelDescOffset
    return $ Texture norm fmt (x,y,z) dsc

  poke p (Texture norm fmt (x,y,z) dsc) = do
    {# set textureReference.normalized #} p (cFromBool norm)
    {# set textureReference.filterMode #} p (cFromEnum fmt)
    pokeArray (p `plusPtr` texRefAddressModeOffset :: Ptr CInt) (map cFromEnum [x,y,z])
    pokeByteOff p texRefChannelDescOffset dsc


--------------------------------------------------------------------------------
-- Texture References
--------------------------------------------------------------------------------

-- |Bind the memory area associated with the device pointer to a texture
-- reference given by the named symbol. Any previously bound references are
-- unbound.
--
{-# INLINEABLE bind #-}
bind :: String -> Texture -> DevicePtr a -> Int64 -> IO ()
bind !name !tex !dptr !bytes = do
  ref <- getTex name
  poke ref tex
  nothingIfOk =<< cudaBindTexture ref dptr (format tex) bytes

{-# INLINE cudaBindTexture #-}
{# fun unsafe cudaBindTexture
  { alloca- `Int'
  , id      `TextureReference'
  , dptr    `DevicePtr a'
  , with_*  `FormatDesc'
  ,         `Int64'            } -> `Status' cToEnum #}
  where dptr = useDevicePtr . castDevPtr

-- |Bind the two-dimensional memory area to the texture reference associated
-- with the given symbol. The size of the area is constrained by (width,height)
-- in texel units, and the row pitch in bytes. Any previously bound references
-- are unbound.
--
{-# INLINEABLE bind2D #-}
bind2D :: String -> Texture -> DevicePtr a -> (Int,Int) -> Int64 -> IO ()
bind2D !name !tex !dptr (!width,!height) !bytes = do
  ref <- getTex name
  poke ref tex
  nothingIfOk =<< cudaBindTexture2D ref dptr (format tex) width height bytes

{-# INLINE cudaBindTexture2D #-}
{# fun unsafe cudaBindTexture2D
  { alloca- `Int'
  , id      `TextureReference'
  , dptr    `DevicePtr a'
  , with_*  `FormatDesc'
  ,         `Int'
  ,         `Int'
  ,         `Int64'             } -> `Status' cToEnum #}
  where dptr = useDevicePtr . castDevPtr


-- |Returns the texture reference associated with the given symbol
--
{-# INLINEABLE getTex #-}
getTex :: String -> IO TextureReference
getTex !name = resultIfOk =<< cudaGetTextureReference name

{-# INLINE cudaGetTextureReference #-}
{# fun unsafe cudaGetTextureReference
  { alloca-       `Ptr Texture' peek*
  , withCString_* `String'            } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

{-# INLINE with_ #-}
with_ :: Storable a => a -> (Ptr a -> IO b) -> IO b
with_ = with


-- CUDA 5.0 changed the types of some attributes from char* to void*
--
{-# INLINE withCString_ #-}
withCString_ :: String -> (Ptr a -> IO b) -> IO b
withCString_ !str !fn = withCString str (fn . castPtr)

