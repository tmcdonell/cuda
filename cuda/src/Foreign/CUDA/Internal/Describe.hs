--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Internal.Describe
-- Copyright : [2016..2023] Trevor L. McDonell
-- License   : BSD
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Internal.Describe
  where


-- | Like 'Text.Show.Show', but focuses on providing a more detailed description
-- of the value rather than a 'Text.Read.read'able representation.
--
class Describe a where
    describe :: a -> String

