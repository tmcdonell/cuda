CUDA_INSTALL_PATH	?= /usr/local/cuda
NVCC			?= $(CUDA_INSTALL_PATH)/bin/nvcc
INCLUDES		+= . $(CUDA_INSTALL_PATH)/include
LIB			+= $(CUDA_INSTALL_PATH)/lib
BUILD   		?= build

MAIN			 = Main
CUFILES			 = scan_kernel.cu
CUOBJS			 = $(addprefix $(BUILD)/,$(addsuffix .o,$(CUFILES)))
NVCCFLAGS		+= $(addprefix -I,$(INCLUDES))

ifeq ($(emu),1)
    NVCCFLAGS 		+= -deviceemu
endif

$(BUILD)/%.cu.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(BUILD)/%.hs : %.chs
	c2hs $(addprefix --cppopts=-I,$(INCLUDES)) --output-dir=$(BUILD) $<

$(MAIN) : mkdirs $(BUILD)/$(MAIN).hs $(CUOBJS)
	ghc --make -lstdc++ -lcudart $(CUOBJS) -odir $(BUILD) -hidir $(BUILD) $(BUILD)/$(MAIN).hs -o $@

mkdirs :
	mkdir -p $(BUILD)

clean:
	$(RM) $(MAIN) $(patsubst %.cu,%.linkinfo,$(CUFILES))
	$(RM) -r $(BUILD)

