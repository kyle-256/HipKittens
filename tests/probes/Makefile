# Build the lane-layout probe(s) as standalone HIP host executables.
# Run on chi2811 (gfx950 / MI355X).

HIPCXX ?= /opt/rocm/bin/hipcc
HIP_INCLUDE_DIR := /opt/rocm/include/hip
THUNDERKITTENS_ROOT ?= $(realpath ../..)

HIPFLAGS = -std=c++20 \
           --offload-arch=gfx950 \
           -DKITTENS_CDNA4 \
           -DHIP_ENABLE_WARP_SYNC_BUILTINS \
           -O2 \
           -w \
           -I$(THUNDERKITTENS_ROOT)/include \
           -I$(HIP_INCLUDE_DIR)

PROBES := rrr_b_lane_layout_probe

all: $(PROBES)

$(PROBES): %: %.cu
	$(HIPCXX) $(HIPFLAGS) $< -o $@

clean:
	rm -f $(PROBES)

.PHONY: all clean
