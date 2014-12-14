{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# OPTIONS_HADDOCK prune #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Texture
-- Copyright : [2009..2014] Trevor L. McDonell
-- License   : BSD
--
-- Texture management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Texture (

  -- * Texture Reference Management
  Texture(..), Format(..), AddressMode(..), FilterMode(..), ReadMode(..),
  create, destroy,
  bind, bind2D,
  getAddressMode, getFilterMode, getFormat,
  setAddressMode, setFilterMode, setFormat, setReadMode,

  -- Internal
  peekTex

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Ptr
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Marshal
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Control.Monad

#if CUDA_VERSION >= 3020
{-# DEPRECATED create, destroy "as of CUDA version 3.2" #-}
#endif


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A texture reference
--
newtype Texture = Texture { useTexture :: {# type CUtexref #}}
  deriving (Eq, Show)

instance Storable Texture where
  sizeOf _    = sizeOf    (undefined :: {# type CUtexref #})
  alignment _ = alignment (undefined :: {# type CUtexref #})
  peek p      = Texture `fmap` peek (castPtr p)
  poke p t    = poke (castPtr p) (useTexture t)

-- |
-- Texture reference addressing modes
--
{# enum CUaddress_mode as AddressMode
  { underscoreToCase }
  with prefix="CU_TR_ADDRESS_MODE" deriving (Eq, Show) #}

-- |
-- Texture reference filtering mode
--
{# enum CUfilter_mode as FilterMode
  { underscoreToCase }
  with prefix="CU_TR_FILTER_MODE" deriving (Eq, Show) #}

-- |
-- Texture read mode options
--
#c
typedef enum CUtexture_flag_enum {
  CU_TEXTURE_FLAG_READ_AS_INTEGER        = CU_TRSF_READ_AS_INTEGER,
  CU_TEXTURE_FLAG_NORMALIZED_COORDINATES = CU_TRSF_NORMALIZED_COORDINATES,
  CU_TEXTURE_FLAG_SRGB                   = CU_TRSF_SRGB
} CUtexture_flag;
#endc

{# enum CUtexture_flag as ReadMode
  { underscoreToCase
  , CU_TEXTURE_FLAG_SRGB as SRGB }
  with prefix="CU_TEXTURE_FLAG" deriving (Eq, Show) #}

-- |
-- Texture data formats
--
{# enum CUarray_format as Format
  { underscoreToCase
  , UNSIGNED_INT8  as Word8
  , UNSIGNED_INT16 as Word16
  , UNSIGNED_INT32 as Word32
  , SIGNED_INT8    as Int8
  , SIGNED_INT16   as Int16
  , SIGNED_INT32   as Int32 }
  with prefix="CU_AD_FORMAT" deriving (Eq, Show) #}


--------------------------------------------------------------------------------
-- Texture management
--------------------------------------------------------------------------------

-- |
-- Create a new texture reference. Once created, the application must call
-- 'setPtr' to associate the reference with allocated memory. Other texture
-- reference functions are used to specify the format and interpretation to be
-- used when the memory is read through this reference.
--
{-# INLINEABLE create #-}
create :: IO Texture
create = resultIfOk =<< cuTexRefCreate

{-# INLINE cuTexRefCreate #-}
{# fun unsafe cuTexRefCreate
  { alloca- `Texture' peekTex* } -> `Status' cToEnum #}


-- |
-- Destroy a texture reference
--
{-# INLINEABLE destroy #-}
destroy :: Texture -> IO ()
destroy !tex = nothingIfOk =<< cuTexRefDestroy tex

{-# INLINE cuTexRefDestroy #-}
{# fun unsafe cuTexRefDestroy
  { useTexture `Texture' } -> `Status' cToEnum #}


-- |
-- Bind a linear array address of the given size (bytes) as a texture
-- reference. Any previously bound references are unbound.
--
{-# INLINEABLE bind #-}
bind :: Texture -> DevicePtr a -> Int64 -> IO ()
bind !tex !dptr !bytes = nothingIfOk =<< cuTexRefSetAddress tex dptr bytes

{-# INLINE cuTexRefSetAddress #-}
{# fun unsafe cuTexRefSetAddress
  { alloca-         `Int'
  , useTexture      `Texture'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int64'       } -> `Status' cToEnum #}


-- |
-- Bind a linear address range to the given texture reference as a
-- two-dimensional arena. Any previously bound reference is unbound. Note that
-- calls to 'setFormat' can not follow a call to 'bind2D' for the same texture
-- reference.
--
{-# INLINEABLE bind2D #-}
bind2D :: Texture -> Format -> Int -> DevicePtr a -> (Int,Int) -> Int64 -> IO ()
bind2D !tex !fmt !chn !dptr (!width,!height) !pitch =
  nothingIfOk =<< cuTexRefSetAddress2DSimple tex fmt chn dptr width height pitch

{-# INLINE cuTexRefSetAddress2DSimple #-}
{# fun unsafe cuTexRefSetAddress2DSimple
  { useTexture      `Texture'
  , cFromEnum       `Format'
  ,                 `Int'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'
  ,                 `Int'
  ,                 `Int64'       } -> `Status' cToEnum #}


-- |
-- Get the addressing mode used by a texture reference, corresponding to the
-- given dimension (currently the only supported dimension values are 0 or 1).
--
{-# INLINEABLE getAddressMode #-}
getAddressMode :: Texture -> Int -> IO AddressMode
getAddressMode !tex !dim = resultIfOk =<< cuTexRefGetAddressMode tex dim

{-# INLINE cuTexRefGetAddressMode #-}
{# fun unsafe cuTexRefGetAddressMode
  { alloca-    `AddressMode' peekEnum*
  , useTexture `Texture'
  ,            `Int'                   } -> `Status' cToEnum #}


-- |
-- Get the filtering mode used by a texture reference
--
{-# INLINEABLE getFilterMode #-}
getFilterMode :: Texture -> IO FilterMode
getFilterMode !tex = resultIfOk =<< cuTexRefGetFilterMode tex

{-# INLINE cuTexRefGetFilterMode #-}
{# fun unsafe cuTexRefGetFilterMode
  { alloca-    `FilterMode' peekEnum*
  , useTexture `Texture'              } -> `Status' cToEnum #}


-- |
-- Get the data format and number of channel components of the bound texture
--
{-# INLINEABLE getFormat #-}
getFormat :: Texture -> IO (Format, Int)
getFormat !tex = do
  (!status,!fmt,!dim) <- cuTexRefGetFormat tex
  resultIfOk (status,(fmt,dim))

{-# INLINE cuTexRefGetFormat #-}
{# fun unsafe cuTexRefGetFormat
  { alloca-    `Format'    peekEnum*
  , alloca-    `Int'       peekIntConv*
  , useTexture `Texture'                } -> `Status' cToEnum #}


-- |
-- Specify the addressing mode for the given dimension of a texture reference
--
{-# INLINEABLE setAddressMode #-}
setAddressMode :: Texture -> Int -> AddressMode -> IO ()
setAddressMode !tex !dim !mode = nothingIfOk =<< cuTexRefSetAddressMode tex dim mode

{-# INLINE cuTexRefSetAddressMode #-}
{# fun unsafe cuTexRefSetAddressMode
  { useTexture `Texture'
  ,            `Int'
  , cFromEnum  `AddressMode' } -> `Status' cToEnum #}


-- |
-- Specify the filtering mode to be used when reading memory through a texture
-- reference
--
{-# INLINEABLE setFilterMode #-}
setFilterMode :: Texture -> FilterMode -> IO ()
setFilterMode !tex !mode = nothingIfOk =<< cuTexRefSetFilterMode tex mode

{-# INLINE cuTexRefSetFilterMode #-}
{# fun unsafe cuTexRefSetFilterMode
  { useTexture `Texture'
  , cFromEnum  `FilterMode' } -> `Status' cToEnum #}


-- |
-- Specify additional characteristics for reading and indexing the texture
-- reference
--
{-# INLINEABLE setReadMode #-}
setReadMode :: Texture -> ReadMode -> IO ()
setReadMode !tex !mode = nothingIfOk =<< cuTexRefSetFlags tex mode

{-# INLINE cuTexRefSetFlags #-}
{# fun unsafe cuTexRefSetFlags
  { useTexture `Texture'
  , cFromEnum  `ReadMode' } -> `Status' cToEnum #}


-- |
-- Specify the format of the data and number of packed components per element to
-- be read by the texture reference
--
{-# INLINEABLE setFormat #-}
setFormat :: Texture -> Format -> Int -> IO ()
setFormat !tex !fmt !chn = nothingIfOk =<< cuTexRefSetFormat tex fmt chn

{-# INLINE cuTexRefSetFormat #-}
{# fun unsafe cuTexRefSetFormat
  { useTexture `Texture'
  , cFromEnum  `Format'
  ,            `Int'     } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

{-# INLINE peekTex #-}
peekTex :: Ptr {# type CUtexref #} -> IO Texture
peekTex = liftM Texture . peek

