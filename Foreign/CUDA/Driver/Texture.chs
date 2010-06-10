{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Texture
-- Copyright : (c) [2009..2010] Trevor L. McDonell
-- License   : BSD
--
-- Texture management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Texture
  (
    Texture(..), AddressMode(..), FilterMode(..), Format,
    create, destroy,
    getPtr, getAddressMode, getFilterMode,
    setPtr, setAddressMode, setFilterMode, setFormat,

    -- Internal
    peekTex
  )
  where

#include <cuda.h>
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


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |A texture reference
--
newtype Texture a = Texture { useTexture :: {# type CUtexref #}}

instance Storable (Texture a) where
  sizeOf _    = sizeOf    (undefined :: {# type CUtexref #})
  alignment _ = alignment (undefined :: {# type CUtexref #})
  peek p      = Texture `fmap` peek (castPtr p)
  poke p t    = poke (castPtr p) (useTexture t)

-- |Texture reference addressing modes
--
{# enum CUaddress_mode as AddressMode
  { underscoreToCase }
  with prefix="CU_TR_ADDRESS_MODE" deriving (Eq, Show) #}

-- |Texture reference filtering mode
--
{# enum CUfilter_mode as FilterMode
  { underscoreToCase }
  with prefix="CU_TR_FILTER_MODE" deriving (Eq, Show) #}

-- |Texture read mode options
--
#c
typedef enum CUtexture_flag_enum {
  CU_TEXTURE_FLAG_READ_AS_INTEGER        = CU_TRSF_READ_AS_INTEGER,
  CU_TEXTURE_FLAG_NORMALIZED_COORDINATES = CU_TRSF_NORMALIZED_COORDINATES
} CUtexture_flag;
#endc

{# enum CUtexture_flag as ReadMode
  { underscoreToCase }
  with prefix="CU_TEXTURE_FLAG" deriving (Eq, Show) #}

-- |Texture data formats
--
{# enum CUarray_format as TextureFormat
  { }
  with prefix="CU_AD_FORMAT" deriving (Eq, Show) #}


class Format a where
  tag      :: a -> Int
  channels :: a -> Int
  channels _ = 1

instance Format Word   where tag _ = fromEnum UNSIGNED_INT32
instance Format Word8  where tag _ = fromEnum UNSIGNED_INT8
instance Format Word16 where tag _ = fromEnum UNSIGNED_INT16
instance Format Word32 where tag _ = fromEnum UNSIGNED_INT32

instance Format Int    where tag _ = fromEnum SIGNED_INT32
instance Format Int8   where tag _ = fromEnum SIGNED_INT8
instance Format Int16  where tag _ = fromEnum SIGNED_INT16
instance Format Int32  where tag _ = fromEnum SIGNED_INT32

instance Format Float  where tag _ = fromEnum FLOAT
instance Format Double where
  tag _      = fromEnum SIGNED_INT32
  channels _ = 2
  -- __hiloint2double()

-- FIXME: half (16-bit float) ??
-- FIXME: vector types ??

--------------------------------------------------------------------------------
-- Texture management
--------------------------------------------------------------------------------

-- |Create a new texture reference. Once created, the application must call
-- 'setPtr' to associate the reference with allocated memory. Other texture
-- reference functions are used to specify the format and interpretation to be
-- used when the memory is read through this reference.
--
create :: Format a => IO (Texture a)
create = do
  tex <- resultIfOk =<< cuTexRefCreate
  setFormat tex
  return tex

{# fun unsafe cuTexRefCreate
  { alloca- `Texture a' peekTex* } -> `Status' cToEnum #}


-- |Destroy a texture reference
--
destroy :: Texture a -> IO ()
destroy tex = nothingIfOk =<< cuTexRefDestroy tex

{# fun unsafe cuTexRefDestroy
  { useTexture `Texture a' } -> `Status' cToEnum #}


-- |Get the address associated with a texture reference
--
getPtr :: Texture a -> IO (DevicePtr a)
getPtr tex = resultIfOk =<< cuTexRefGetAddress tex

{# fun unsafe cuTexRefGetAddress
  { alloca-    `DevicePtr a' peekDevPtr*
  , useTexture `Texture a'               } -> `Status' cToEnum #}


-- |Get the addressing mode used by a texture reference, corresponding to the
-- given dimension (currently the only supported dimension values are 0 or 1).
--
getAddressMode :: Texture a -> Int -> IO AddressMode
getAddressMode tex dim = resultIfOk =<< cuTexRefGetAddressMode tex dim

{# fun unsafe cuTexRefGetAddressMode
  { alloca-    `AddressMode' peekEnum*
  , useTexture `Texture a'
  ,            `Int'                  } -> `Status' cToEnum #}


-- |Get the filtering mode used by a texture reference
--
getFilterMode :: Texture a -> IO FilterMode
getFilterMode tex = resultIfOk =<< cuTexRefGetFilterMode tex

{# fun unsafe cuTexRefGetFilterMode
  { alloca-    `FilterMode' peekEnum*
  , useTexture `Texture a'            } -> `Status' cToEnum #}

{-
-- |Get the data format and number of channel components of the bound texture
--
getFormat :: Texture a -> IO (Format, Int)
getFormat tex = do
  (status,fmt,dim) <- cuTexRefGetFormat tex
  resultIfOk (status,(fmt,dim))

{# fun unsafe cuTexRefGetFormat
  { alloca-    `Format'    peekEnum*
  , alloca-    `Int'       peekIntConv*
  , useTexture `Texture a'              } -> `Status' cToEnum #}
-}

-- |Bind a linear array address of the given size (bytes) as a texture
-- reference. Any previously bound references are unbound.
--
setPtr :: Texture a -> DevicePtr a -> Int -> IO ()
setPtr tex dptr bytes = nothingIfOk =<< cuTexRefSetAddress tex dptr bytes

{# fun unsafe cuTexRefSetAddress
  { alloca-         `Int'
  , useTexture      `Texture a'
  , useDeviceHandle `DevicePtr a'
  ,                 `Int'         } -> `Status' cToEnum #}


-- |Specify the addressing mode for the given dimension of a texture reference
--
setAddressMode :: Texture a -> Int -> AddressMode -> IO ()
setAddressMode tex dim mode = nothingIfOk =<< cuTexRefSetAddressMode tex dim mode

{# fun unsafe cuTexRefSetAddressMode
  { useTexture `Texture a'
  ,            `Int'
  , cFromEnum  `AddressMode' } -> `Status' cToEnum #}


-- |Specify the filtering mode to be used when reading memory through a texture
-- reference
--
setFilterMode :: Texture a -> FilterMode -> IO ()
setFilterMode tex mode = nothingIfOk =<< cuTexRefSetFilterMode tex mode

{# fun unsafe cuTexRefSetFilterMode
  { useTexture `Texture a'
  , cFromEnum  `FilterMode' } -> `Status' cToEnum #}


-- |Specify the format of the data to be read by the texture reference
--
setFormat :: Format a => Texture a -> IO ()
setFormat tex = doSet undefined tex
  where
    doSet :: Format b => b -> Texture b -> IO ()
    doSet fmt _ = nothingIfOk =<< cuTexRefSetFormat tex (tag fmt) (channels fmt)

{# fun unsafe cuTexRefSetFormat
  { useTexture `Texture a'
  ,            `Int'
  ,            `Int'       } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

peekTex :: Ptr {# type CUtexref #} -> IO (Texture a)
peekTex = liftM Texture . peek

