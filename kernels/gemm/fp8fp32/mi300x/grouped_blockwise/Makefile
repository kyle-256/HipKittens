# Grouped FP8 blockwise GEMM (fwd + wgrad) — MI300X gfx942.
#
#   make                                 # default tk_kernel.so (fwd)
#   make wgrad                           # default tk_wgrad.so
#   make tuned BLOCK_M=128 ... CHUNK=8   # per-shape tuned fwd .so
#   make wgrad-tuned BLOCK_M=128 ...     # per-shape tuned wgrad .so

GPU_TARGET ?= CDNA3
THUNDERKITTENS_ROOT ?= $(abspath ../../../../../)
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIPCXX           ?= $(ROCM_INSTALL_DIR)/bin/hipcc

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

all: tk_kernel

tk_kernel: grouped_blockwise.cpp
	$(HIPCXX) grouped_blockwise.cpp $(HIPFLAGS) $(ICXXFLAGS) \
	    -DTK_MODULE_NAME=tk_kernel -o tk_kernel$(EXT) 2>&1

wgrad: grouped_wgrad.cpp
	$(HIPCXX) grouped_wgrad.cpp $(HIPFLAGS) $(ICXXFLAGS) \
	    -DTK_MODULE_NAME=tk_wgrad -o tk_wgrad$(EXT) 2>&1

# Per-shape tuned fwd builds: produces tk_kernel_BM<…>_…so picked by test harness.
BLOCK_M   ?= 256
BLOCK_N   ?= 128
BLOCK_K   ?= 128
NUM_WARPS ?= 8
REG_M     ?= 64
CHUNK     ?= 4

TUNED_NAME := tk_kernel_BM$(BLOCK_M)_BN$(BLOCK_N)_BK$(BLOCK_K)_W$(NUM_WARPS)
ifneq ($(REG_M),64)
TUNED_NAME := $(TUNED_NAME)_RM$(REG_M)
endif
ifneq ($(CHUNK),4)
TUNED_NAME := $(TUNED_NAME)_CHK$(CHUNK)
endif

tuned: grouped_blockwise.cpp
	$(HIPCXX) grouped_blockwise.cpp $(HIPFLAGS) $(ICXXFLAGS) \
	    -DBLOCK_M=$(BLOCK_M) -DBLOCK_N=$(BLOCK_N) -DBLOCK_K=$(BLOCK_K) \
	    -DNUM_WARPS=$(NUM_WARPS) -DREG_M_BUILD=$(REG_M) -DCHIPLET_CHUNK=$(CHUNK) \
	    -DTK_MODULE_NAME=$(TUNED_NAME) \
	    -o $(TUNED_NAME)$(EXT) 2>&1

# Tuned wgrad variant.
WGRAD_TUNED_NAME := tk_wgrad_BM$(BLOCK_M)_BN$(BLOCK_N)_BK$(BLOCK_K)_W$(NUM_WARPS)
wgrad-tuned: grouped_wgrad.cpp
	$(HIPCXX) grouped_wgrad.cpp $(HIPFLAGS) $(ICXXFLAGS) \
	    -DBLOCK_M=$(BLOCK_M) -DBLOCK_N=$(BLOCK_N) -DBLOCK_K=$(BLOCK_K) -DNUM_WARPS=$(NUM_WARPS) \
	    -DTK_MODULE_NAME=$(WGRAD_TUNED_NAME) \
	    -o $(WGRAD_TUNED_NAME)$(EXT) 2>&1

clean:
	rm -f tk_kernel$(EXT) tk_wgrad$(EXT)

clean-tuned:
	rm -f tk_kernel_BM*$(EXT) tk_wgrad_BM*$(EXT)
