#
# Common bridge between Haskell and CUDA build systems
#
# For the variables set below, it was required to edit the common.mk file from
# the SDK to be optional defines (?=).
#

# ------------------------------------------------------------------------------
# Common CUDA build system
# ------------------------------------------------------------------------------

CUDA_SDK_PATH	?= /Developer/CUDA

SRCDIR		?= ./
DISTROOT	?= dist
BINDIR		 = $(DISTROOT)/bin
ROOTOBJDIR	 = $(DISTROOT)/build
LIBDIR		 = $(CUDA_SDK_PATH)/lib
COMMONDIR	 = $(CUDA_SDK_PATH)/common

GHC_FLAGS	 = --make -Wall -lstdc++ \
		   -i$(SRCDIR) -i$(ROOTOBJDIR)/$(BINSUBDIR) \
		   -odir $(ROOTOBJDIR)/$(BINSUBDIR) -hidir $(ROOTOBJDIR)/$(BINSUBDIR)

HSFILES		+= $(HSMAIN)

# small hack: dependencies of the target, but not passed to object linker
#
CUBINS		+= $(patsubst %.chs,$(OBJDIR)/%.hs,$(notdir $(filter %.chs,$(HSFILES))))
CUBINS		+= $(patsubst %.hsc,$(OBJDIR)/%.hs,$(notdir $(filter %.hsc,$(HSFILES))))

ifeq ($(dbg),1)
    GHC_FLAGS	+= -prof -auto-all -caf-all -fhpc
else
    GHC_FLAGS	+= -O3
endif

ifeq ($(suffix $(HSMAIN)),.hs)
    LINK	 = ghc $(GHC_FLAGS) $(SRCDIR)$(HSMAIN)
else
    LINK	 = ghc $(GHC_FLAGS) $(OBJDIR)/$(addsuffix .hs,$(basename $(HSMAIN)))
endif

# ------------------------------------------------------------------------------
# Rules
# ------------------------------------------------------------------------------
include $(COMMONDIR)/common.mk

$(OBJDIR)/%.hs : $(SRCDIR)%.chs $(SRCDIR)C2HS.hs
	$(VERBOSE)c2hs --include=$(OBJDIR) $(addprefix --cppopts=,$(INCLUDES)) --output-dir=$(OBJDIR) --output=$(notdir $@) $<

$(OBJDIR)/%.hs : $(SRCDIR)%.hsc
	$(VERBOSE)hsc2hs $(INCLUDES) -o $@ $<

$(SRCDIR)C2HS.hs :
	$(VERBOSE)c2hs --output-dir=$(SRCDIR) --copy-library

spotless : clean
	$(VERBOSE)rm -rf $(DISTROOT)
	$(VERBOSE)rm -f  $(patsubst %.cu,%.linkinfo,$(notdir $(CUFILES)))

