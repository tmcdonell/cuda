--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Analysis
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Meta-module exporting CUDA analysis routines
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Analysis (module Analysis)
  where

import Foreign.CUDA.Analysis.Device     as Analysis
import Foreign.CUDA.Analysis.Occupancy  as Analysis

