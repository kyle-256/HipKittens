# FP8 blockwise GEMM (fwd/dgrad/wgrad) — MI300X gfx942.
#
#   make                                              # default tk_kernel.so
#   make tuned BLOCK_M=128 NUM_WARPS=4 BW_RAW_DRAIN=1 # per-shape tuned .so

GPU_TARGET ?= CDNA3
SRC        := blockwise.cpp
TARGET     := tk_kernel

THUNDERKITTENS_ROOT ?= $(abspath ../../../../../)
ROCM_INSTALL_DIR    := $(ROCM_PATH)
HIPCXX              ?= $(ROCM_INSTALL_DIR)/bin/hipcc

ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS += -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
endif

ICXXFLAGS := -std=c++20 -w -shared -fPIC -O3 -ffast-math \
             -Rpass-analysis=kernel-resource-usage \
             -I$(THUNDERKITTENS_ROOT)/include -I$(THUNDERKITTENS_ROOT)/prototype \
             -I$(ROCM_INSTALL_DIR)/include -I$(ROCM_INSTALL_DIR)/include/hip \
             $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags)

EXT := $(shell python3-config --extension-suffix)

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) -o $(TARGET)$(EXT) 2>&1

# Per-shape tuned builds — produces a uniquely-named .so per
# (BM, BN, BK, NW, REG_M, RAW, PRE, CHK) combo.
# test_python.py:TUNED_REGISTRY picks the right .so per shape.
BLOCK_M          ?= 256
BLOCK_N          ?= 128
BLOCK_K          ?= 128
NUM_WARPS        ?= 8
REG_M            ?= 64
BW_RAW_DRAIN     ?= 0
BW_PRESCALE_BS   ?= 0
BW_CHIPLET_CHUNK ?= 1

TUNED_NAME := tk_kernel_BM$(BLOCK_M)_BN$(BLOCK_N)_BK$(BLOCK_K)_W$(NUM_WARPS)
ifneq ($(REG_M),64)
TUNED_NAME := $(TUNED_NAME)_RM$(REG_M)
endif
ifneq ($(BW_RAW_DRAIN),0)
TUNED_NAME := $(TUNED_NAME)_RAW$(BW_RAW_DRAIN)
endif
ifneq ($(BW_PRESCALE_BS),0)
TUNED_NAME := $(TUNED_NAME)_PRE$(BW_PRESCALE_BS)
endif
ifneq ($(BW_CHIPLET_CHUNK),1)
TUNED_NAME := $(TUNED_NAME)_CHK$(BW_CHIPLET_CHUNK)
endif

TUNED_FLAGS := -DBLOCK_M=$(BLOCK_M) -DBLOCK_N=$(BLOCK_N) -DBLOCK_K=$(BLOCK_K) \
               -DNUM_WARPS=$(NUM_WARPS) -DREG_M_BUILD=$(REG_M) \
               -DBW_RAW_DRAIN=$(BW_RAW_DRAIN) -DBW_PRESCALE_BS=$(BW_PRESCALE_BS) \
               -DBW_CHIPLET_CHUNK=$(BW_CHIPLET_CHUNK) \
               -DTK_MODULE_NAME=$(TUNED_NAME)

tuned: $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(TUNED_FLAGS) \
	    -o $(TUNED_NAME)$(EXT) 2>&1

clean:
	rm -f $(TARGET)$(EXT)

clean-tuned:
	rm -f tk_kernel_BM*_BN*_BK*_W*$(EXT)
