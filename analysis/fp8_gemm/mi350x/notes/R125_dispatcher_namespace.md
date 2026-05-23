# R125 — Dispatcher namespace strategy

v2 lives in `namespace hk_fp8_kernel_v2` (anonymous namespace wrap in PT adapter).
v1 lives in `namespace hk_fp8_kernel` (same anonymous wrap, different name).

Each TU includes both via PT adapter `hk_grouped_gemm_gfx950.cu`:
```cpp
namespace { 
  namespace hk_fp8_kernel { #include "kernel_fp8_layouts.cpp" }
  namespace hk_fp8_kernel_v2 { #include "kernel_fp8_layouts2.cpp" }
}
```

v2 file pulls v1 helpers via `#include "kernel_fp8_layouts.cpp"`. v2 namespace gets a copy of all v1 symbols inside it.

Risk: type name collisions when defining new types in v2 (R60 issue with A_row_reg_32 vs A_row_reg). New types must use unique names + always within `v2_pinned` sub-namespace.

For multi-session: standardize all new v2 types as `v2_pinned::TypeName_v2` to avoid future collisions.
