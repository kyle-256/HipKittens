# R130 — Build cache hazard (per [[fp8-rrr-grad-a-fix-2026-05-19]])

When changing v2 fragment types or kernel templates:
1. `build/temp/csrc/kernels/grouped_gemm/HipKittens/*.o` may be stale (template instances)
2. PT _hip.cpp mirrors are .gitignored, must remove manually
3. `.so` linked from cached `.o` may have wrong symbols
4. Always `rm build/lib/libprimus_turbo_kernels.so` + corresponding `.o` before fragment-related changes

For pure dispatcher / wrapper additions: rebuild usually works.
For new template instantiations: nuke build cache.

This session avoided this hazard (foundation probes are new symbols, not changes to existing). Multi-session P1.2 will hit it (replacing existing template body).
