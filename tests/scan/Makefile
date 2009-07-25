CUDA_INSTALL_PATH	?= /usr/local/cuda
NVCC			?= $(CUDA_INSTALL_PATH)/bin/nvcc
INCLUDES		+= $(CUDA_INSTALL_PATH)/include
LIB			+= $(CUDA_INSTALL_PATH)/lib

TARGET			 = Main
CUFILES			 = scan_kernel.cu
CUOBJS			 = $(addsuffix .o,$(CUFILES))
NVCCFLAGS		+= $(addprefix -I,$(INCLUDES))

ifeq ($(emu),1)
    NVCCFLAGS 		+= -deviceemu
endif

%.cu.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.hs : %.chs
	c2hs $(addprefix --cppopts=-I,$(INCLUDES)) $<

$(TARGET) : $(TARGET).hs $(CUOBJS)
	ghc --make -lstdc++ -lcudart $(CUOBJS) $<

clean:
	$(RM) $(TARGET) $(CUOBJS) $(patsubst %.cu,%.linkinfo,$(CUFILES))
	$(RM) $(TARGET).{hi,chi,chs.h,hs,o}

