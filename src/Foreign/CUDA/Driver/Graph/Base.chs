{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
#ifdef USE_EMPTY_CASE
{-# LANGUAGE EmptyCase                #-}
#endif
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Graph.Base
-- Copyright : [2018] Trevor L. McDonell
-- License   : BSD
--
-- Graph execution functions for the low-level driver interface
--
-- Requires CUDA-10
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Graph.Base
  where

#include "cbits/stubs.h"
{# context lib="cuda" #}


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

newtype Graph = Graph { useGraph :: {# type CUgraph #}}
  deriving (Eq, Show)

data GraphFlag
instance Enum GraphFlag where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("GraphFlag.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif

newtype Node = Node { useNode :: {# type CUgraphNode #}}
  deriving (Eq, Show)

{# enum CUgraphNodeType
  { underscoreToCase
  , CU_GRAPH_NODE_TYPE_GRAPH as Subgraph
  }
  with prefix="CU_GRAPH_NODE_TYPE" deriving (Eq, Show, Bounded) #}

newtype Executable = Executable { useExecutable :: {# type CUgraphExec #}}
  deriving (Eq, Show)

