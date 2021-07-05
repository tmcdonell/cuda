--------------------------------------------------------------------------------
-- |
-- Module    : Text.Show.Describe
-- Copyright : [2016..2020] Trevor L. McDonell
-- License   : BSD
--
--------------------------------------------------------------------------------

module Text.Show.Describe
  where


-- | Like 'Text.Show.Show', but focuses on providing a more detailed description
-- of the value rather than a 'Text.Read.read'able representation.
--
class Describe a where
    describe :: a -> String

