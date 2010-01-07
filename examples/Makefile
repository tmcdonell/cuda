ifneq ($(emu),1)
    PROJECTS := $(shell find src -name Makefile)
else
    PROJECTS := $(shell find src -name Makefile | xargs grep -L 'USEDRVAPI')
endif


%.do :
	$(MAKE) -C $(dir $*) $(MAKECMDGOALS)

all : $(addsuffix .do,$(PROJECTS))
	@echo "Finished building all"

clean : $(addsuffix .do,$(PROJECTS))
	@echo "Finished cleaning all"

clobber : $(addsuffix .do,$(PROJECTS))
	@echo "Finished cleaning all"
