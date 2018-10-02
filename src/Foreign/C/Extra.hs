{-# LANGUAGE CPP                      #-}
{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.C.Extra
-- Copyright : [2018] Trevor L. McDonell
-- License   : BSD
--
--------------------------------------------------------------------------------

module Foreign.C.Extra (

  c_strnlen,

) where

import Foreign.C


#if defined(WIN32)
{-# INLINE c_strnlen' #-}
c_strnlen' :: CString -> CSize -> IO CSize
c_strnlen' str size = do
  str' <- peekCStringLen (str, fromIntegral size)
  return $ stringLen 0 str'
  where
    stringLen acc []       = acc
    stringLen acc ('\0':_) = acc
    stringLen acc (_:xs)   = stringLen (acc+1) xs
#else
foreign import ccall unsafe "string.h strnlen" c_strnlen'
  :: CString -> CSize -> IO CSize
#endif

{-# INLINE c_strnlen #-}
c_strnlen :: CString -> Int -> IO Int
c_strnlen str maxlen = fromIntegral `fmap` c_strnlen' str (fromIntegral maxlen)

