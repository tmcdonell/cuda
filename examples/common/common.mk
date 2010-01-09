# ------------------------------------------------------------------------------
#
# Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and 
# proprietary rights in and to this software and related documentation. 
# Any use, reproduction, disclosure, or distribution of this software 
# and related documentation without an express license agreement from
# NVIDIA Corporation is strictly prohibited.
#
# Please refer to the applicable NVIDIA end user license agreement (EULA) 
# associated with this source code for terms and conditions that govern 
# your use of this NVIDIA software.
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Common Haskell/CUDA build system
# ------------------------------------------------------------------------------

.SUFFIXES : .cu .cu_dbg.o .c_dbg.o .cpp_dbg.o .cu_rel.o .c_rel.o .cpp_rel.o .cubin .ptx

# No CUDA compatible device present
# MAIN_DEVICE     := $(shell ghc -e "Control.Monad.liftM (Data.Either.either id Foreign.CUDA.deviceName) (Foreign.CUDA.props 0)")
# ifeq ($(MAIN_DEVICE),"Device Emulation (CPU)")
#     emu		:= 1
# endif

# Add new SM Versions here as devices with new Compute Capability are released
SM_VERSIONS     := sm_10 sm_11 sm_12 sm_13

# detect OS
OSUPPER         := $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER         := $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
DARWIN          := $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
    SNOWLEOPARD := $(strip $(findstring 10.6, $(shell egrep "<string>10\.6.*</string>" /System/Library/CoreServices/SystemVersion.plist)))
endif

# detect if 32 bit or 64 bit system
HP_64           := $(strip $(shell uname -m | grep 64))

# Basic directory setup
CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_SDK_PATH     ?= /Developer/GPU\ Computing/C

SRCDIR          ?= .
ROOTDIR         ?= ..
ROOTBINDIR      := $(ROOTDIR)/../bin
BINDIR		:= $(ROOTBINDIR)/$(OSLOWER)
ROOTOBJDIR      := obj
LIBDIR          := $(ROOTDIR)/../lib
COMMONDIR	:= $(ROOTDIR)/../common

# Compilers
NVCC            := $(CUDA_INSTALL_PATH)/bin/nvcc
GHC             := ghc
C2HS            := c2hs
HSC2HS          := hsc2hs
CXX             := g++
CC              := gcc
LINK            := g++ -fPIC

# Includes
INCLUDES        += -I. -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/include -I$(COMMONDIR)/src

# architecture flag for cubin build
CUBIN_ARCH_FLAG :=

# Warning flags
CXXWARN_FLAGS := \
        -W -Wall \
        -Wimplicit \
        -Wswitch \
        -Wformat \
        -Wchar-subscripts \
        -Wparentheses \
        -Wmultichar \
        -Wtrigraphs \
        -Wpointer-arith \
        -Wcast-align \
        -Wreturn-type \
        -Wno-unused-function

CWARN_FLAGS := $(CXXWARN_FLAGS) \
        -Wstrict-prototypes \
        -Wmissing-prototypes \
        -Wmissing-declarations \
        -Wnested-externs \
        -Wmain

GHCWARN_FLAGS := \
        -Wall

# cross-compilation flags
ifeq ($(x86_64),1)
    NVCCFLAGS	+= -m64
    ifneq ($(DARWIN),)
        CXX_ARCH_FLAGS 	+= -arch x86_64
    else
        CXX_ARCH_FLAGS	+= -m64
    endif
else
ifeq ($(i386),1)
    NVCCFLAGS	+= -m32
    ifneq ($(DARWIN),)
        CXX_ARCH_FLAGS	+= -arch i386
    else
        CXX_ARCH_FLAGS	+= -m32
    endif
else
    ifneq ($(SNOWLEOPARD),)
        NVCCFLAGS	+= -m32
        CXX_ARCH_FLAGS	+= -arch i386 -m32
    endif
endif
endif

# Compiler-specific flags
NVCCFLAGS       +=
GHCFLAGS        += $(GHCWARN_FLAGS) -i$(SRCDIR) -i$(COMMONDIR)/src -i$(OBJDIR) -odir $(OBJDIR) -hidir $(OBJDIR) --make
CXXFLAGS        += $(CXXWARN_FLAGS) $(CXX_ARCH_FLAGS)
CFLAGS          += $(CWARN_FLAGS) $(CXX_ARCH_FLAGS)
LINK		+= $(CXX_ARCH_FLAGS)

# Common flags
COMMONFLAGS     += $(INCLUDES) -DUNIX

# Debug/release configuration
ifeq ($(dbg),1)
    COMMONFLAGS += -g
    GHCFLAGS	+= -prof -auto-all -fhpc
    NVCCFLAGS   += -D_DEBUG
    CXXFLAGS    += -D_DEBUG
    CFLAGS      += -D_DEBUG
    BINSUBDIR   := debug
    LIBSUFFIX   := D
else
    COMMONFLAGS += -O2
    GHCFLAGS	+= -O2
    BINSUBDIR   := release
    LIBSUFFIX   :=
    NVCCFLAGS   += --compiler-options -fno-strict-aliasing
    CXXFLAGS    += -fno-strict-aliasing
    CFLAGS      += -fno-strict-aliasing
endif

# Device emulation configuration
ifeq ($(emu),1)
    NVCCFLAGS   += -deviceemu
    CUDACCFLAGS +=
    BINSUBDIR   := emu$(BINSUBDIR)
    LIBSUFFIX   := $(LIBSUFFIX)_emu
    # consistency, makes developing easier
    C2HSFLAGS   += --cppopts=-D__DEVICE_EMULATION__
    GHCFLAGS    += -D__DEVICE_EMULATION__
    CXXFLAGS    += -D__DEVICE_EMULATION__
    CFLAGS      += -D__DEVICE_EMULATION__
endif

# architecture flag for cubin build
CUBIN_ARCH_FLAG :=

# Libraries
LIB             += -L$(LIBDIR) $(addprefix -l,$(EXTRALIBS))
ifeq ($(HP_64),)
   LIB          += -L$(CUDA_INSTALL_PATH)/lib
else
   LIB          += -L$(CUDA_INSTALL_PATH)/lib64
endif

# If dynamically linking to CUDA and CUDART, we exclude the libraries from the LIB
ifeq ($(USECUDADYNLIB),1)
    LIB         += -ldl -rdynamic
else
    # static linking, we will statically link against CUDA and CUDART
    ifeq ($(USEDRVAPI),1)
        LIB     += -lcuda
    else
        LIB     += -lcudart
    endif
endif

ifeq ($(USECUFFT),1)
    ifeq ($(emu),1)
        LIB     += -lcufftemu
    else
        LIB     += -lcufft
    endif
endif

ifeq ($(USECUBLAS),1)
    ifeq ($(emu),1)
        LIB     += -lcublasemu
    else
        LIB     += -lcublas
    endif
endif

ifeq ($(USECUDPP),1)
    CUDPPLIB := cudpp

    ifneq ($(HP_64),)
        CUDPPLIB := $(CUDPPLIB)64
    endif

    ifeq ($(emu),1)
        CUDPPLIB := $(CUDPPLIB)_emu
    endif

    LIB     += -l$(CUDPPLIB)
endif

# Library/executable configuration
ifneq ($(STATIC_LIB),)
    TARGETDIR   := $(LIBDIR)
    TARGET      := $(LIBDIR)/$(basename $(STATIC_LIB))$(LIBSUFFIX)$(suffix $(STATIC_LIB))
    LINKLINE     = ar rucv $(TARGET) $(OBJS); ranlib $(TARGET)
else
ifneq ($(DYNAMIC_LIB),)
    TARGETDIR   := $(LIBDIR)
    TARGET      := $(LIBDIR)/$(basename $(DYNAMIC_LIB))$(LIBSUFFIX)$(suffix $(DYNAMIC_LIB))
    CFLAGS      += -fPIC
    CXXFLAGS    += -fPIC
    NVCCFLAGS   += -Xcompiler -fPIC
    ifneq ($(DARWIN),)
    LINKLINE     = $(LINK) -dynamiclib -o $(TARGET) -install_name "@rpath/$(notdir $(TARGET))" $(OBJS) $(LIB)
    else
    LINKLINE     = $(LINK) -shared -o $(TARGET) -Wl,-rpath,$(notdir $(TARGET)) $(OBJS) $(LIB)
    endif
else
    TARGETDIR   := $(BINDIR)/$(BINSUBDIR)
    TARGET      := $(TARGETDIR)/$(EXECUTABLE)
    ifneq ($(HSMAIN),)
        ifeq ($(suffix $(HSMAIN)),.chs)
            FOO      := $(HSMAIN)
            CHSFILES += $(FOO)
            HSMAIN    = $(OBJDIR)/$(basename $(FOO)).hs
        endif

        ifeq ($(dbg),1)
#            OBJS     += $(OBJDIR)/ptxvars.cu.o
            LINKLINE  = $(GHC) -o $(TARGET) $(LIB) $(OBJS) $(GHCFLAGS) $(HSMAIN)
        else
            LINKLINE  = $(GHC) -o $(TARGET) $(LIB) $(OBJS) $(GHCFLAGS) $(HSMAIN)
        endif
    else
        LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB)
    endif
endif
endif

# check if verbose
ifeq ($(verbose),1)
    VERBOSE     :=
else
    VERBOSE     := @
endif


# ------------------------------------------------------------------------------
# Check for input flags and set compiler flags appropriately
# ------------------------------------------------------------------------------
ifeq ($(fastmath),1)
    NVCCFLAGS   += -use_fast_math
endif

ifeq ($(keep),1)
    NVCCFLAGS       += -keep
    NVCC_KEEP_CLEAN := *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx
endif

ifdef maxregisters
    NVCCFLAGS   += -maxrregcount $(maxregisters)
endif

# Add cudacc flags
NVCCFLAGS += $(CUDACCFLAGS)

# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)

ifeq ($(nvcc_warn_verbose),1)
    NVCCFLAGS += $(addprefix --compiler-options ,$(CXXWARN_FLAGS))
    NVCCFLAGS += --compiler-options -fno-strict-aliasing
endif


# ------------------------------------------------------------------------------
# Set up object files
# ------------------------------------------------------------------------------
OBJDIR  := $(ROOTOBJDIR)/$(BINSUBDIR)
OBJS    += $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(CCFILES)))
OBJS    += $(patsubst %.c,$(OBJDIR)/%.c.o,$(notdir $(CFILES)))
OBJS    += $(patsubst %.cu,$(OBJDIR)/%.cu.o,$(notdir $(CUFILES)))

# ------------------------------------------------------------------------------
# Set up preprocessed Haskell files
# ------------------------------------------------------------------------------
DEPS	+= $(patsubst %.chs,$(OBJDIR)/%.hs,$(notdir $(CHSFILES)))
DEPS	+= $(patsubst %.hsc,$(OBJDIR)/%.hs,$(notdir $(HSCFILES)))

# ------------------------------------------------------------------------------
# Set up cubin output files
# ------------------------------------------------------------------------------
CUBINDIR := $(SRCDIR)/data
CUBINS   += $(patsubst %.cu,$(CUBINDIR)/%.cubin,$(notdir $(CUBINFILES)))

# ------------------------------------------------------------------------------
# Set up PTX output files
# ------------------------------------------------------------------------------
PTXDIR  := $(SRCDIR)/data
PTXBINS += $(patsubst %.cu,$(PTXDIR)/%.ptx,$(notdir $(PTXFILES)))


# ------------------------------------------------------------------------------
# Rules
# ------------------------------------------------------------------------------
default: $(TARGET)

%.subdir :
	$(VERBOSE)$(MAKE) -C $* $(MAKECMDGOALS)

$(OBJDIR)/%.c.o : $(SRCDIR)/%.c $(C_DEPS)
	$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp.o : $(SRCDIR)/%.cpp $(C_DEPS)
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/ptxvars.cu.o: makedirectories
	$(VERBOSE)$(NVCC) -g -G --host-compilation=C -D__DEVICE_LAUNCH_PARAMETERS_H__ -Xptxas -fext -o $@ -c $(CUDA_INSTALL_PATH)/bin/ptxvars.cu

$(OBJDIR)/%.cu.o : $(SRCDIR)/%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<

$(OBJDIR)/%.hs : $(SRCDIR)/%.chs
	$(VERBOSE)$(C2HS) $(C2HSFLAGS) --include=$(OBJDIR) $(addprefix --cppopts=,$(INCLUDES)) --output-dir=$(OBJDIR) --output=$(notdir $@) $<

$(OBJDIR)/%.hs : $(SRCDIR)/%.hsc
	$(VERBOSE)$(HSC2HS) $(INCLUDES) -o $@ $<

$(CUBINDIR)/%.cubin : $(SRCDIR)/%.cu cubindirectory
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -cubin $<

$(PTXDIR)/%.ptx : $(SRCDIR)/%.cu ptxdirectory
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -ptx $<

#
# The following definition is a template that gets instantiated for each SM
# version (sm_10, sm_13, etc.) stored in SMVERSIONS.  It does 2 things:
# 1. It adds to OBJS a .cu_sm_XX.o for each .cu file it finds in CUFILES_sm_XX.
# 2. It generates a rule for building .cu_sm_XX.o files from the corresponding
#    .cu file.
#
# The intended use for this is to allow Makefiles that use common.mk to compile
# files to different Compute Capability targets (aka SM arch version).  To do
# so, in the Makefile, list files for each SM arch separately, like so:
#
# CUFILES_sm_10 := mycudakernel_sm10.cu app.cu
# CUFILES_sm_12 := anothercudakernel_sm12.cu
#
define SMVERSION_template
OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu_$(1).o,$(notdir $(CUFILES_$(1))))
$(OBJDIR)/%.cu_$(1).o : $(SRCDIR)/%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) -o $$@ -c $$< $(NVCCFLAGS) -arch $(1)
endef

# This line invokes the above template for each arch version stored in
# SM_VERSIONS.  The call funtion invokes the template, and the eval
# function interprets it as make commands.
$(foreach smver,$(SM_VERSIONS),$(eval $(call SMVERSION_template,$(smver))))

$(TARGET): makedirectories $(DEPS) $(OBJS) $(CUBINS) $(PTXBINS) Makefile $(addsuffix .subdir,$(SUBDIRS))
	$(VERBOSE)$(LINKLINE)

cubindirectory:
	$(VERBOSE)mkdir -p $(CUBINDIR)

ptxdirectory:
	$(VERBOSE)mkdir -p $(PTXDIR)

makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)
	$(VERBOSE)mkdir -p $(TARGETDIR)


tidy : $(addsuffix .subdir,$(SUBDIRS))
	$(VERBOSE)find . | egrep "#"  | xargs rm -f
	$(VERBOSE)find . | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f $(OBJS)
	$(VERBOSE)rm -f $(CUBINS)
	$(VERBOSE)rm -f $(PTXBINS)
	$(VERBOSE)rm -f $(TARGET)
	$(VERBOSE)rm -f $(NVCC_KEEP_CLEAN)
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.ppm
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.pgm
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.bin
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.bmp

clobber : clean
	$(VERBOSE)rm -rf $(ROOTOBJDIR)

spotless:
	$(VERBOSE)rm -rf .hpc
	$(VERBOSE)rm -f $(EXECUTABLE).{aux,hp,prof,ps}
	$(VERBOSE)rm -f *.html
	$(VERBOSE)find . -name "*.tix" -print0 | xargs -0 rm -f

