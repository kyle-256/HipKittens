
# Usage:
#   make                    # builds all default kernels
#   (previously: make SRC=my_kernel.cpp TARGET=my_kernel)
#
# Example kernels built by default:
#   tk_kernel_bkwd       from attn_bkwd_causal.cpp
#   tk_kernel_bkwd_prep  from attn_bkwd_prep.cpp
#   tk_kernel_fwd        from attn_fwd_causal.cpp

# Compiler
GPU_TARGET ?= CDNA4

# parameters
ATTN_B ?= 16
ATTN_H ?= 64
ATTN_H_KV ?= 8
ATTN_N ?= 4096

CPPFLAGS += -DATTN_B=$(ATTN_B)
CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DATTN_H_KV=$(ATTN_H_KV)
CPPFLAGS += -DATTN_N=$(ATTN_N)

# HIP variables
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),CDNA2)
  HIPFLAGS+= -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
  HIPFLAGS+= -DKITTENS_CDNA3 --offload-arch=gfx942
else ifeq ($(GPU_TARGET),CDNA4)
  HIPFLAGS+= -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
endif

# Common variables and flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS ?= -Wall -Wextra
CXXFLAGS := -w

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
ICXXFLAGS += $(PY_LDFLAGS)
ICXXFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype \
             $(shell python3 -m pybind11 --includes) -shared -fPIC -Rpass-analysis=kernel-resource-usage

# Log directory & file (one per make invocation)
LOGDIR := /workdir/data_logs/$(shell date +%m%d_%H%M%S)_outputs
LOGFILE := $(LOGDIR)/make_build.log

# Asymmetric head dimensions (for Llama 4 style Q_dim=192, V_dim=128)
ATTN_D_QK ?= 0
ATTN_D_V ?= 0

# --------------------------------------------------------------------
# Default target: build all three kernels (symmetric D=128)
# --------------------------------------------------------------------
all: tk_kernel_bkwd tk_kernel_bkwd_prep tk_kernel_fwd

# Asymmetric target: build D_QK=192 / D_V=128 kernels (prep + fwd + bkwd)
asymmetric: tk_kernel_bkwd_prep tk_kernel_fwd tk_kernel_bkwd
	@echo "Asymmetric D_QK=192 D_V=128 kernels built (prep + fwd + bkwd)"

# --------------------------------------------------------------------
# Select prep / fwd / bkwd source based on asymmetric flag (must precede rules
# that use PREP_SRC / FWD_SRC / BKWD_SRC in prerequisites)
# --------------------------------------------------------------------
# Asymmetric MLA (D_QK=192, D_V=128): you MUST pass e.g. ATTN_D_QK=192 so BKWD_SRC
# selects attn_bkwd_causal_d192v128.cpp. If ATTN_D_QK is unset, this block is skipped
# and the symmetric attn_bkwd_causal.cpp is used instead.
ifneq ($(ATTN_D_QK), 0)
PREP_SRC=attn_bkwd_prep_d192v128.cpp
FWD_SRC=attn_fwd_causal_d192v128.cpp
BKWD_SRC=attn_bkwd_causal_d192v128.cpp
else
PREP_SRC=attn_bkwd_prep.cpp
FWD_SRC=attn_fwd_causal.cpp
BKWD_SRC=attn_bkwd_causal.cpp
endif

# --------------------------------------------------------------------
# Explicit rules for each kernel
# --------------------------------------------------------------------

tk_kernel_bkwd: $(BKWD_SRC)
	@mkdir -p $(LOGDIR)
	$(HIPCXX) $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o $@$(shell python3-config --extension-suffix) 2>&1 | tee $(LOGFILE)

tk_kernel_bkwd_prep: $(PREP_SRC)
	@mkdir -p $(LOGDIR)
	$(HIPCXX) $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o $@$(shell python3-config --extension-suffix) 2>&1 | tee $(LOGFILE)

tk_kernel_fwd: $(FWD_SRC)
	@mkdir -p $(LOGDIR)
	$(HIPCXX) $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o $@$(shell python3-config --extension-suffix) 2>&1 | tee $(LOGFILE)

# Clean target
clean:
	rm -f tk_kernel_bkwd* tk_kernel_bkwd_prep* tk_kernel_fwd*

.PHONY: all asymmetric clean

