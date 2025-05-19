SHELL=/bin/bash -o pipefail
.SHELLFLAGS += -e

PKG_SRCS  :=
HDL_SRCS  := $(shell find $(PWD)/../hdl -name '*.sv')
COMM_HVL  := $(shell find $(PWD)/../hvl/common                -name '*.sv' -o -name '*.v' -o -name '*.o' -o -name '*.so')
VCS_HVL   := $(COMM_HVL) $(shell find $(PWD)/../hvl/vcs       -name '*.sv' -o -name '*.v' -o -name '*.o' -o -name '*.so')
VER_HVL   := $(COMM_HVL) $(shell find $(PWD)/../hvl/verilator -name '*.sv' -o -name '*.v' -o -name '*.o' -o -name '*.so' -o -name '*.cpp')
SRAM_SRCS :=
HDRS      := $(shell find $(PWD)/../hvl -name '*.svh')
DW_IP     :=
VCS_SRCS  := $(PKG_SRCS) $(HDL_SRCS) $(VCS_HVL) $(SRAM_SRCS) $(DW_IP)
VER_SRCS  := $(PKG_SRCS) $(HDL_SRCS) $(VER_HVL) $(SRAM_SRCS) $(DW_IP)

export VCS_ARCH_OVERRIDE=linux
COMMON_FLAGS = +incdir+$(PWD)/../hvl/common
VCS_FLAGS    = $(COMMON_FLAGS) +incdir+$(PWD)/../hvl/vcs -licqueue -full64 -lca -sverilog -timescale=1ps/1ps -debug_acc+all -kdb -suppress=LCA_FEATURES_ENABLED -msg_config=../vcs_warn.config -xprop=../xprop.config -xprop=flowctrl -assert svaext
VER_FLAGS    = $(COMMON_FLAGS) +incdir+$(PWD)/../hvl/verilator -Wall --timescale 1ps/1ps -Mdir build -O3 -CFLAGS "-Ofast -march=native" --x-assign fast --x-initial fast --noassert --cc --exe ../verilator_warn.vlt --trace-fst --trace-structs --trace-max-array 1048576 --trace-max-width 1048576 +define+ECE411_VERILATOR

vcs/top_tb: $(VCS_SRCS) $(HDRS)
	mkdir -p vcs
	python3 check_sus.py
	cd vcs && vcs $(VCS_SRCS) $(VCS_FLAGS) -l compile.log -top top_tb -o top_tb
	bash check_compile_error.sh

.PHONY: run_vcs_top_tb
run_vcs_top_tb: vcs/top_tb
	rm -f vcs/dump.fsdb
	cd vcs && ./top_tb -l simulation.log -exitstatus -suppress=ASLR_DETECTED_INFO \
		+CLOCK_PERIOD_PS_ECE411=500

verilator/build/Vtop_tb: $(VER_SRCS) $(HDRS)
	mkdir -p verilator/build
	python3 check_sus.py
	cd verilator ;\
	verilator $(VER_FLAGS) $(VER_SRCS) --top-module top_tb |& tee build/compile.log
	cd verilator/build ;\
	$(MAKE) --jobs=$(shell echo $(shell nproc)-2 | bc) -f Vtop_tb.mk |& tee -a compile.log

.PHONY: run_verilator_top_tb
run_verilator_top_tb: verilator/build/Vtop_tb
	find ./verilator -maxdepth 1 -type f -delete
	cd verilator && ./build/Vtop_tb \
		+CLOCK_PERIOD_PS_ECE411=500 \
		|& tee simulation.log

.PHONY: verdi
verdi:
	mkdir -p verdi
	cd verdi && timeout $(ECE411_GUI_TIMEOUT) $(VERDI_HOME)/bin/verdi -ssf $(PWD)/vcs/dump.fsdb

.PHONY: clean
clean:
	rm -rf bin vcs verdi verilator
