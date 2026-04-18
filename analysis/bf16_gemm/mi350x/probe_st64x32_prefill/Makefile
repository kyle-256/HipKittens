# P23 Dev B — LDS-peek probe for st_64x32_padded_b128.
# Builds a standalone HIP binary (no pybind, no JIT) that links the TK
# header-only library and exercises the new shape via prefill_swizzled_offsets
# + load + LDS readback.

TK_ROOT ?= /shared_nfs/kyle/HipKittens2

HIPCXX  ?= /opt/rocm/bin/hipcc
HIPFLAGS := -DKITTENS_CDNA4 --offload-arch=gfx950 \
            -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math \
            -I/opt/rocm/include/rocrand
INCLUDES := -I$(TK_ROOT)/include -I$(TK_ROOT)/prototype -I/opt/rocm/include/hip
CXXFLAGS := -std=c++20 -w -O2

probe: probe.cpp
	$(HIPCXX) probe.cpp $(HIPFLAGS) $(CXXFLAGS) $(INCLUDES) -o probe

clean:
	rm -f probe

.PHONY: clean
