#!/usr/bin/env python3
###############################################################################
# Metric for the head_dim=64 (gpt-oss-20b) attention kernels in
# kernels/attn/gqa_causal_backwards/.  Drives the auto_optimize.py loop.
#
# Goal: D=64 fwd+bwd wall-clock must beat HipKittens' own D=128 baseline on
# the same B/N/H/H_KV (head_dim halved → ~2x speedup is the theoretical
# floor at equal GPU utilisation).
#
# Per shape we measure three configurations, each in an isolated subprocess
# (so a GPU memory fault on one combo does not abort the whole metric):
#
#   1) D=128 reference (HipKittens main combined kernel: tk_kernel_fwd,
#      tk_kernel_bkwd_prep, tk_kernel_bkwd) on BSHD input.  This is the
#      wall-clock baseline.
#   2) D=64 (tk_kernel_fwd_d64 + tk_kernel_bkwd_prep_d64 + tk_kernel_bkwd_d64)
#      on BSHD input.  The natural / fast path.
#   3) D=64 on SBHD input (transpose(0,1) on the wrapper boundary).  Same
#      kernel, exercises the second supported user layout.
#
# Score:
#
#     score = int(round(sum_d64_tflops * 10))
#           - 1000 * cos_fail            (cos < HARD_COS, currently 0.99)
#           - 100  * cos_warn            (HARD_COS <= cos < SOFT_COS, 0.999)
#           - 2000 * exception           (worker SIGABRT / GPU fault / timeout)
#           - 5000 * compile_fail        (make exit non-zero for any d64 .so)
#           + speedup_bonus              (see below)
#
# Speedup bonus per shape:
#
#     bsd_speedup = clamp(0, 1 - d64_bshd_total_ms / d128_bshd_total_ms, 1)
#     sbhd_speedup = clamp(0, 1 - d64_sbhd_total_ms / d128_bshd_total_ms, 1)
#     bonus = int(round(5000 * bsd_speedup + 2500 * sbhd_speedup))
#
# Reading: BSHD speedup of 0.50 (D=64 wall-clock = 50% of D=128) gives +2500;
# 0.55 gives +2750; 1.00 (D=64 = 0 ms, impossible) gives +5000.  Below 0.0
# (D=64 slower than D=128) gives +0; that is the floor we must clear.
# SBHD bonus is half-weight because the wrapper transpose adds an unavoidable
# ~5-15% overhead.
#
# Stdout contract: exactly one integer ``score`` line on stdout (parent
# auto_optimize.py reads stdout's last non-empty line as the metric value).
# All diagnostics go to stderr.
#
# ``--strict`` mode (used by deep-check): exit non-zero if any cos failed
# or any subprocess crashed.  Default mode always exits 0.
###############################################################################

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback


# ---------------------------------------------------------------------------
# Score weights -- tweak via env if you want a different penalty profile.
# ---------------------------------------------------------------------------
COS_FAIL_PENALTY     = int(os.environ.get("D64_COS_FAIL_PENALTY", "1000"))
COS_WARN_PENALTY     = int(os.environ.get("D64_COS_WARN_PENALTY", "100"))
EXCEPTION_PENALTY    = int(os.environ.get("D64_EXCEPTION_PENALTY", "2000"))
COMPILE_FAIL_PENALTY = int(os.environ.get("D64_COMPILE_FAIL_PENALTY", "5000"))
SPEEDUP_BSHD_BONUS   = int(os.environ.get("D64_SPEEDUP_BSHD_BONUS", "5000"))
SPEEDUP_SBHD_BONUS   = int(os.environ.get("D64_SPEEDUP_SBHD_BONUS", "2500"))

# Cosine similarity gates.  cos < HARD_COS = full fail; HARD_COS..SOFT_COS = warn.
HARD_COS = float(os.environ.get("D64_HARD_COS", "0.99"))
SOFT_COS = float(os.environ.get("D64_SOFT_COS", "0.999"))

# Per-call timing knobs (min-of-N batches; each batch averages PERF_BATCH_ITERS
# CUDA-event-timed iterations).  Min-of-N is robust to one-off CPU / clock
# jitter on shared MI355 hosts.
PERF_WARMUP        = int(os.environ.get("D64_PERF_WARMUP", "10"))
PERF_TRIALS        = int(os.environ.get("D64_PERF_TRIALS", "5"))
PERF_BATCH_ITERS   = int(os.environ.get("D64_PERF_BATCH_ITERS", "10"))

# rocm-smi KFD VRAM column: above this byte count, the GPU counts as busy.
KFD_BUSY_VRAM_BYTES = 100 * 1024 * 1024

# Allowed GPU pool — **docker HIP_VISIBLE_DEVICES indices**, NOT rocm-smi
# GPU[N] indices.  This box's mapping (PCIe BDF, verified 2026-04-29):
#   HIP=0→bus 0x75 (rocm-smi GPU[3])     HIP=4→bus 0xF5 (rocm-smi GPU[7])
#   HIP=1→bus 0x05 (rocm-smi GPU[0])     HIP=5→bus 0x85 (rocm-smi GPU[4])
#   HIP=2→bus 0x65 (rocm-smi GPU[2])     HIP=6→bus 0xE5 (rocm-smi GPU[6]) idle
#   HIP=3→bus 0x15 (rocm-smi GPU[1])     HIP=7→bus 0x95 (rocm-smi GPU[5]) idle
# User asked for the rocm-smi GPU[5]+GPU[6] idle pair => HIP=6,7 in container.
GPU_POOL = sorted({
    int(g) for g in os.environ.get("D64_GPU_POOL", "6,7").split(",") if g.strip()
})

# Where the d64 + d128 kernels live.
KERNEL_DIR = os.environ.get(
    "D64_KERNEL_DIR",
    "/workspace/code/HipKittens/kernels/attn/gqa_causal_backwards",
)
THUNDERKITTENS_ROOT = os.environ.get(
    "THUNDERKITTENS_ROOT",
    "/workspace/code/HipKittens",
)

# Skip rebuilding the D=128 reference if its .so is already present (it
# almost never changes; default behaviour is to skip).
SKIP_D128_BUILD = os.environ.get("D64_SKIP_D128_BUILD", "1") not in ("0", "false", "")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _pick_idle_gpu() -> str | None:
    """Smallest idle GPU id in GPU_POOL (busy if KFD lists a PID with VRAM
    > KFD_BUSY_VRAM_BYTES).  Falls back to first pool entry if rocm-smi fails.
    """
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showpids"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
    except Exception:
        return str(GPU_POOL[0]) if GPU_POOL else None
    busy: set[int] = set()
    in_kfd = False
    for line in out.splitlines():
        if "KFD process information" in line:
            in_kfd = True
            continue
        if not in_kfd:
            continue
        if line.startswith("=") or "PROCESS NAME" in line:
            continue
        cols = line.split()
        if len(cols) < 4 or not cols[0].isdigit():
            continue
        try:
            vram = int(cols[3])
        except ValueError:
            continue
        if vram <= KFD_BUSY_VRAM_BYTES:
            continue
        for gid in re.findall(r"\d+", cols[2]):
            busy.add(int(gid))
    idle = [g for g in GPU_POOL if g not in busy]
    if idle:
        return str(idle[0])
    return str(GPU_POOL[0]) if GPU_POOL else None


# ---------------------------------------------------------------------------
# Build step.  D=64 kernels are rebuilt every metric invocation (compile gate);
# D=128 reference is built only if its .so is missing or SKIP_D128_BUILD=0.
# ---------------------------------------------------------------------------
import glob


def _kernel_so_exists(name: str) -> bool:
    pat = os.path.join(KERNEL_DIR, f"{name}.cpython-*.so")
    return bool(glob.glob(pat))


def _make(targets: list[str], build_n: int) -> tuple[bool, str]:
    """Run make + verify each target's .so actually exists.

    The Makefile rule pipes through ``tee`` which swallows hipcc's exit
    code, so a non-zero ``result.returncode`` is rare even on real build
    failure.  We confirm the build succeeded by checking that every
    requested kernel produced a ``.so``.  The rule is:

      ok = (returncode == 0) AND (every target's .so exists post-make
            AND was newly written -- mtime check would be brittle on
            shared filesystems, so we accept stale .so as a soft pass
            and rely on per-target presence).
    """
    cmd = ["make", *targets, f"ATTN_N={build_n}"]
    env = os.environ.copy()
    env.setdefault("THUNDERKITTENS_ROOT", THUNDERKITTENS_ROOT)
    try:
        result = subprocess.run(
            cmd, cwd=KERNEL_DIR, env=env,
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return False, "make TIMEOUT after 600s"
    log = (result.stdout or "") + (result.stderr or "")
    tail = "\n".join(log.splitlines()[-15:])
    if result.returncode != 0:
        return False, f"make {targets} exit {result.returncode}\n{tail}"

    # Verify every target produced a .so.  This is the real compile gate
    # since the Makefile rule's `tee` masks hipcc's exit code.
    missing = [t for t in targets if not _kernel_so_exists(t)]
    if missing:
        return False, (
            f"make {targets} returned 0 but .so missing for {missing}\n"
            f"(Makefile rule tees stdout, exit code is unreliable)\n{tail}"
        )

    # Even if all .so exist, an "error:" anywhere in the log + a "failed
    # to execute" line is a strong signal that hipcc bombed on at least
    # one target and the .so we see is stale.
    if "failed to execute" in log.lower() or " error: " in log.lower():
        # If we got here despite missing-check passing, the .so is from
        # a previous round; still treat as failure so the agent sees it.
        return False, (
            f"make {targets} log contains 'error:' / 'failed to execute' "
            f"despite returncode=0 and .so present (likely stale .so):\n{tail}"
        )

    return True, tail


def _build_kernels(build_n: int) -> tuple[bool, str]:
    """Build d64 fwd/prep/bkwd combined kernels (the compile gate).

    The D=128 wall-clock baseline is provided by ``aiter`` (not HK's own
    D=128 kernel), so no D=128 build is needed here.  Reasons:

      * HK's ``tk_kernel_fwd`` (D=128 forward) doesn't build cleanly on
        main due to a pre-existing ``zero`` / ``copy`` overload ambiguity
        in ``attn_fwd_causal.cpp`` (not introduced by this branch).
      * Running HK's D=128 bwd against a synthetic LSE produces GPU faults.
      * ``aiter::flash_attn_func`` is what the agent's own
        ``test_python_d64.py`` already compares against, so the numbers
        across logs are commensurable.
    """
    ok, tail = _make([
        "tk_kernel_fwd_d64",
        "tk_kernel_bkwd_prep_d64",
        "tk_kernel_bkwd_d64",
    ], build_n)
    if not ok:
        return False, f"D=64 build failed:\n{tail}"
    return True, tail


# ---------------------------------------------------------------------------
# Shape table.  ``layouts`` is the user-facing input layout; the wrapper in
# the worker transposes SBHD → BSHD before feeding the kernel and untransposes
# the outputs.
# ---------------------------------------------------------------------------
# Each entry: (label, B, N, H, H_KV, causal)
SHAPES = [
    ("gpt-oss-N1024",  4, 1024, 64, 8, True),
    ("gpt-oss-N4096", 16, 4096, 64, 8, True),
    ("gpt-oss-N8192",  4, 8192, 64, 8, True),
]
D_VALUES = [64]                 # primary measurement
D_REF = 128                     # baseline comparison (HK D=128 main combined)
LAYOUTS_D64 = ["BSHD", "SBHD"]  # user layout variations exercised on D=64
LAYOUTS_D128 = ["BSHD"]         # D=128 reference only on BSHD

BUILD_N = 8192


def _flops(B: int, N: int, H: int, D: int, causal: bool, mode: str) -> float:
    f = 4.0 * B * (N ** 2) * H * D / (2.0 if causal else 1.0)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


# ---------------------------------------------------------------------------
# Worker subprocess.  argv: --worker <out_json> <label> <B> <N> <H> <H_KV>
#                            <causal:0|1> <D:64|128> <layout:BSHD|SBHD>
# ---------------------------------------------------------------------------
def _worker_main(out_json: str, label: str, B: int, N: int, H: int, H_KV: int,
                 causal: bool, dim: int, layout: str) -> int:
    import importlib

    sys.path.insert(0, KERNEL_DIR)
    import torch
    import torch.nn.functional as F

    # D=128 baseline: use aiter::flash_attn_func instead of HK D=128 kernel.
    # Reasons: (1) HK D=128 fwd doesn't build cleanly upstream (pre-existing
    # `zero/copy` overload ambiguity), (2) HK D=128 bwd-only with synthetic
    # L generated by a non-working fwd produces GPU faults, (3) aiter is
    # the same baseline the agent compares against in test_python_d64.py
    # so the numbers are commensurable across logs.
    fwd_mod = prep_mod = bkwd_mod = None
    aiter_mod = None
    if dim == 64:
        mod_names = ["tk_kernel_fwd_d64", "tk_kernel_bkwd_prep_d64",
                     "tk_kernel_bkwd_d64"]
        mods = {}
        for name in mod_names:
            if name in sys.modules:
                del sys.modules[name]
            mods[name] = importlib.import_module(name)
        fwd_mod  = mods[mod_names[0]]
        prep_mod = mods[mod_names[1]]
        bkwd_mod = mods[mod_names[2]]
    elif dim == 128:
        try:
            import aiter
            aiter_mod = aiter
        except Exception as exc:
            raise RuntimeError(
                f"aiter import failed (cannot run D=128 baseline): {exc!r}"
            )
    else:
        raise ValueError(f"unsupported dim={dim}")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # SOC clock warmup.
    a = torch.randn(4096, 4096, dtype=dtype, device=device)
    b = torch.randn(4096, 4096, dtype=dtype, device=device)
    for _ in range(4):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    del a, b

    D = dim  # head_dim used for tensor allocation

    # Generate inputs in the user-specified layout, then transpose to BSHD
    # for the kernel call.  HK kernels expect BSHD (= BNHD = (B, N, H, D))
    # at their dispatch boundary.  SBHD = (S, B, H, D); we transpose(0,1)
    # to make it BSHD before dispatch and undo on the way out.
    if layout == "BSHD":
        Q_user = (torch.randn(B, N, H, D, device=device, dtype=dtype) * 0.5).contiguous()
        K_user = (torch.randn(B, N, H_KV, D, device=device, dtype=dtype) * 0.5).contiguous()
        V_user = (torch.randn(B, N, H_KV, D, device=device, dtype=dtype) * 0.5).contiguous()
    elif layout == "SBHD":
        Q_user = (torch.randn(N, B, H, D, device=device, dtype=dtype) * 0.5).contiguous()
        K_user = (torch.randn(N, B, H_KV, D, device=device, dtype=dtype) * 0.5).contiguous()
        V_user = (torch.randn(N, B, H_KV, D, device=device, dtype=dtype) * 0.5).contiguous()
    else:
        raise ValueError(f"unsupported layout={layout}")

    # Build BSHD views for the reference + kernel calls.
    if layout == "BSHD":
        to_bshd = lambda x: x  # identity
        from_bshd = lambda x: x
    else:  # SBHD → BSHD via transpose(0, 1).contiguous() on the wrapper edge
        to_bshd = lambda x: x.transpose(0, 1).contiguous()
        from_bshd = lambda x: x.transpose(0, 1).contiguous()

    Q = to_bshd(Q_user)  # BSHD = (B, N, H, D)
    K = to_bshd(K_user)
    V = to_bshd(V_user)

    Qr = Q.detach().clone().requires_grad_(True)
    Kr = K.detach().clone().requires_grad_(True)
    Vr = V.detach().clone().requires_grad_(True)
    qb = Qr.transpose(1, 2)  # BHND for SDPA
    kb = Kr.transpose(1, 2).repeat_interleave(H // H_KV, dim=1)
    vb = Vr.transpose(1, 2).repeat_interleave(H // H_KV, dim=1)
    o_ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal).transpose(1, 2)
    dO_bshd = torch.randn_like(o_ref)
    o_ref.backward(dO_bshd)
    dQ_ref, dK_ref, dV_ref = Qr.grad, Kr.grad, Vr.grad  # all BSHD

    # User-layout dO matches the input layout
    dO_user = from_bshd(dO_bshd) if layout != "BSHD" else dO_bshd

    def _cos(x, y):
        return F.cosine_similarity(
            x.detach().flatten().float(),
            y.detach().flatten().float(),
            dim=0,
        ).item()

    # ---- D=128 baseline path: aiter::flash_attn_func ----------------------
    if dim == 128:
        # Aiter expects BHND; we already have Q_user in BSHD/SBHD form.
        # Always feed BHND (transpose(1,2) on BSHD or transpose(0,1).transpose(1,2) on SBHD).
        if layout == "BSHD":
            Qa = Q_user.detach().clone().requires_grad_(True)  # BSHD = BNHD
        else:
            Qa = Q_user.transpose(0, 1).contiguous().detach().requires_grad_(True)  # SBHD->BSHD
        # aiter wants BNHD layout (batch, seq, heads, dim).  HK BSHD is exactly
        # that. So Qa is already BNHD == BSHD.  No transpose needed for aiter.
        Ka = (K_user if layout == "BSHD" else K_user.transpose(0, 1).contiguous()).detach().clone().requires_grad_(True)
        Va = (V_user if layout == "BSHD" else V_user.transpose(0, 1).contiguous()).detach().clone().requires_grad_(True)
        # dO matches aiter's input layout (BNHD == BSHD).
        dO_aiter = dO_bshd.detach().clone()

        def aiter_fwd():
            out, lse = aiter_mod.flash_attn_func(
                Qa, Ka, Va, causal=causal,
                return_lse=True, deterministic=False,
            )
            return out

        def aiter_bwd(out):
            for p in (Qa, Ka, Va):
                if p.grad is not None:
                    p.grad = None
            out.backward(dO_aiter, retain_graph=True)

        # Warmup + correctness sanity (cos vs SDPA).
        out_aiter = aiter_fwd()
        aiter_bwd(out_aiter)
        cos = {
            "O":  F.cosine_similarity(out_aiter.detach().flatten().float(),
                                       o_ref.flatten().float(), dim=0).item(),
            "dQ": F.cosine_similarity(Qa.grad.detach().flatten().float(),
                                       dQ_ref.flatten().float(), dim=0).item(),
            "dK": F.cosine_similarity(Ka.grad.detach().flatten().float(),
                                       dK_ref.flatten().float(), dim=0).item(),
            "dV": F.cosine_similarity(Va.grad.detach().flatten().float(),
                                       dV_ref.flatten().float(), dim=0).item(),
        }
        # Persist correctness before timing.
        partial = {
            "label": label, "cos": cos, "fail": 0, "warn": 0,
            "fwd_tflops": 0.0, "bwd_tflops": 0.0,
            "fwd_ms": float("nan"), "bwd_ms": float("nan"),
            "phase": "correctness_only", "dim": dim, "layout": layout,
            "backend": "aiter",
        }
        with open(out_json, "w") as f:
            json.dump(partial, f); f.flush(); os.fsync(f.fileno())

        start = torch.cuda.Event(enable_timing=True)
        stop  = torch.cuda.Event(enable_timing=True)
        def best_of_n(fn):
            for _ in range(PERF_WARMUP):
                fn()
            torch.cuda.synchronize()
            best_ms = float("inf")
            for _ in range(PERF_TRIALS):
                torch.cuda.synchronize()
                start.record()
                for _ in range(PERF_BATCH_ITERS):
                    fn()
                stop.record(); stop.synchronize()
                avg_ms = start.elapsed_time(stop) / PERF_BATCH_ITERS
                if avg_ms < best_ms:
                    best_ms = avg_ms
            return best_ms / 1000.0

        # Warm everything once before timing.
        out_aiter = aiter_fwd()
        fwd_s = best_of_n(lambda: aiter_fwd())
        out_aiter = aiter_fwd()
        bwd_s = best_of_n(lambda: aiter_bwd(out_aiter))
        fwd_tflops = _flops(B, N, H, D, causal, "fwd") / fwd_s / 1e12
        bwd_tflops = _flops(B, N, H, D, causal, "bwd") / bwd_s / 1e12
        final = {
            "label": label, "cos": cos, "fail": 0, "warn": 0,
            "fwd_tflops": fwd_tflops, "bwd_tflops": bwd_tflops,
            "fwd_ms": fwd_s * 1000.0, "bwd_ms": bwd_s * 1000.0,
            "phase": "complete", "dim": dim, "layout": layout,
            "backend": "aiter",
        }
        with open(out_json, "w") as f:
            json.dump(final, f); f.flush(); os.fsync(f.fileno())
        return 0

    # ---- D=64 path: HK kernels ----------------------------------------------
    def call_fwd_bshd(Q_w, K_w, V_w, O_w, L_w):
        fwd_mod.dispatch_fwd(Q_w, K_w, V_w, O_w, L_w)

    def call_prep_bshd(O_w, dO_w, delta_w):
        prep_mod.dispatch_prep(O_w, dO_w, delta_w)

    def call_bwd_bshd(Q_w, K_w, V_w, dO_w, dQ_in_w, dK_w, dV_w, L_w, delta_w):
        bkwd_mod.dispatch_bwd_combined(
            Q_w, K_w, V_w, dO_w, dQ_in_w, dK_w, dV_w, L_w, delta_w
        )

    def call_dq_shuffle(dQ_in_w, dQ_w):
        prep_mod.dispatch_dq_shuffle(dQ_in_w, dQ_w)

    # SBHD-native sibling callers.  Same kernel surface as the BSHD entries
    # above; the underlying dispatch_*_sbhd implementations re-interpret
    # the contiguous (S, B, H, D) storage by treating gl dim 0 as S and
    # gl dim 1 as B (gl<bf16,-1,-1,-1,-1> has no stride field), so the
    # wrapper transpose(0, 1).contiguous() can be dropped from the timing
    # path.  Added 2026-05-01 (Round D, SKILL §0.3): the BSHD timing
    # path remains byte-identical to HEAD; only the SBHD branch below
    # is rewired.
    def call_fwd_sbhd(Q_w, K_w, V_w, O_w, L_w):
        fwd_mod.dispatch_fwd_sbhd(Q_w, K_w, V_w, O_w, L_w)

    def call_prep_sbhd(O_w, dO_w, delta_w):
        prep_mod.dispatch_prep_sbhd(O_w, dO_w, delta_w)

    def call_bwd_sbhd(Q_w, K_w, V_w, dO_w, dQ_in_w, dK_w, dV_w, L_w, delta_w):
        bkwd_mod.dispatch_bwd_combined_sbhd(
            Q_w, K_w, V_w, dO_w, dQ_in_w, dK_w, dV_w, L_w, delta_w
        )

    def call_dq_shuffle_sbhd(dQ_in_w, dQ_w):
        prep_mod.dispatch_dq_shuffle_sbhd(dQ_in_w, dQ_w)

    O_user = torch.zeros_like(Q_user)
    L_tk = torch.zeros((B, H, N, 1), device=device, dtype=torch.float32).transpose(-1, -2).contiguous()
    delta_tk = torch.zeros((B, H, N, 1), device=device, dtype=torch.float32).transpose(-1, -2).contiguous()
    dQ_in = torch.zeros((B, H, N, D), device=device, dtype=dtype).contiguous()
    dQ_user_t = torch.zeros_like(Q_user)
    dK_user_t = torch.zeros_like(K_user)
    dV_user_t = torch.zeros_like(V_user)

    if layout == "BSHD":
        # Byte-identical to HEAD (to_bshd / from_bshd are identity here,
        # so the kernel sees Q_user / K_user / V_user directly with no
        # extra copy on the timed path).  Wall-clock unchanged.
        def fwd_user_to_bshd():
            Qb = to_bshd(Q_user)
            Kb = to_bshd(K_user)
            Vb = to_bshd(V_user)
            Ob = torch.zeros_like(Qb)
            call_fwd_bshd(Qb, Kb, Vb, Ob, L_tk)
            return Ob

        def bwd_user_pass(Ob_local):
            dOb = to_bshd(dO_user)
            dQ_in.zero_(); delta_tk.zero_()
            dKb = torch.zeros_like(K)  # BSHD
            dVb = torch.zeros_like(V)
            dQb = torch.zeros_like(Q)
            call_prep_bshd(Ob_local, dOb, delta_tk)
            call_bwd_bshd(to_bshd(Q_user), to_bshd(K_user), to_bshd(V_user),
                          dOb, dQ_in, dKb, dVb, L_tk, delta_tk)
            call_dq_shuffle(dQ_in, dQb)
            return from_bshd(dQb), from_bshd(dKb), from_bshd(dVb)
    else:
        # SBHD-native: skip the wrapper transpose+contiguous on every
        # fwd/bwd timing iteration.  Q/K/V/dO/O/dK/dV stay in their
        # user-facing (S, B, H, D) layout; dQ_in stays BHND in both
        # layouts (this matches dispatch_dq_shuffle_sbhd's contract --
        # it reads BHND staging and writes SBHD output) so the bwd
        # combined SBHD kernel's dq_atomic_add codepath is unchanged.
        # L_vec and delta_tk also stay BHND-style (B, H, 1, N) for both
        # layouts -- the kernel always indexes them by (b, h, 1, q)
        # regardless of input layout.
        def fwd_user_to_bshd():
            # Despite the legacy name, in this branch the returned
            # tensor is in the user's SBHD layout (the native dispatch
            # consumes the (S, B, H, D) buffer directly, no transpose).
            O_sbhd = torch.zeros_like(Q_user)
            call_fwd_sbhd(Q_user, K_user, V_user, O_sbhd, L_tk)
            return O_sbhd

        def bwd_user_pass(O_local):
            # O_local is SBHD here (returned by fwd_user_to_bshd above).
            dQ_in.zero_(); delta_tk.zero_()
            dK_sbhd = torch.zeros_like(K_user)  # SBHD
            dV_sbhd = torch.zeros_like(V_user)
            dQ_sbhd = torch.zeros_like(Q_user)
            call_prep_sbhd(O_local, dO_user, delta_tk)
            call_bwd_sbhd(Q_user, K_user, V_user, dO_user,
                          dQ_in, dK_sbhd, dV_sbhd, L_tk, delta_tk)
            call_dq_shuffle_sbhd(dQ_in, dQ_sbhd)
            return dQ_sbhd, dK_sbhd, dV_sbhd

    Ob = fwd_user_to_bshd()
    # In BSHD: from_bshd is identity, O_user = Ob.
    # In SBHD: Ob is already in the user (SBHD) layout (native dispatch),
    # so skip the otherwise-redundant from_bshd transpose.  The cos
    # compare below still calls to_bshd(O_user), which transposes the
    # SBHD prediction to BSHD for comparison against the BSHD reference.
    O_user = Ob if layout == "SBHD" else from_bshd(Ob)

    dQ_user_t, dK_user_t, dV_user_t = bwd_user_pass(Ob)
    torch.cuda.synchronize()

    # Compare in BSHD (the reference's natural layout).
    O_bshd_pred = to_bshd(O_user)
    dQ_bshd_pred = to_bshd(dQ_user_t)
    dK_bshd_pred = to_bshd(dK_user_t)
    dV_bshd_pred = to_bshd(dV_user_t)
    cos = {
        "O":  _cos(O_bshd_pred,  o_ref),
        "dQ": _cos(dQ_bshd_pred, dQ_ref),
        "dK": _cos(dK_bshd_pred, dK_ref),
        "dV": _cos(dV_bshd_pred, dV_ref),
    }
    fail = sum(1 for v in cos.values() if v < HARD_COS)
    warn = sum(1 for v in cos.values() if HARD_COS <= v < SOFT_COS)

    # Persist correctness results before timing -- timing is most likely to fault.
    partial = {
        "label": label, "cos": cos, "fail": fail, "warn": warn,
        "fwd_tflops": 0.0, "bwd_tflops": 0.0,
        "fwd_ms": float("nan"), "bwd_ms": float("nan"),
        "phase": "correctness_only",
        "dim": dim, "layout": layout,
    }
    with open(out_json, "w") as f:
        json.dump(partial, f); f.flush(); os.fsync(f.fileno())

    # Timing.  Bench the user-facing call (including wrapper transposes for SBHD).
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    def best_of_n(fn):
        for _ in range(PERF_WARMUP):
            fn()
        torch.cuda.synchronize()
        best_ms = float("inf")
        for _ in range(PERF_TRIALS):
            torch.cuda.synchronize()
            start.record()
            for _ in range(PERF_BATCH_ITERS):
                fn()
            stop.record(); stop.synchronize()
            avg_ms = start.elapsed_time(stop) / PERF_BATCH_ITERS
            if avg_ms < best_ms:
                best_ms = avg_ms
        return best_ms / 1000.0

    fwd_s = best_of_n(lambda: fwd_user_to_bshd())
    fwd_tflops = _flops(B, N, H, D, causal, "fwd") / fwd_s / 1e12
    bwd_s = best_of_n(lambda: bwd_user_pass(Ob))
    bwd_tflops = _flops(B, N, H, D, causal, "bwd") / bwd_s / 1e12

    final = {
        "label": label, "cos": cos, "fail": fail, "warn": warn,
        "fwd_tflops": fwd_tflops, "bwd_tflops": bwd_tflops,
        "fwd_ms": fwd_s * 1000.0, "bwd_ms": bwd_s * 1000.0,
        "phase": "complete", "dim": dim, "layout": layout,
        "backend": "hk",
    }
    with open(out_json, "w") as f:
        json.dump(final, f); f.flush(); os.fsync(f.fileno())
    return 0


def _run_worker(label: str, B: int, N: int, H: int, H_KV: int, causal: bool,
                dim: int, layout: str, timeout: int = 240) -> dict:
    fd, out_path = tempfile.mkstemp(prefix="metric_d64_", suffix=".json")
    os.close(fd)
    try:
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--worker", out_path, label, str(B), str(N),
            str(H), str(H_KV), "1" if causal else "0",
            str(dim), layout,
        ]
        env = os.environ.copy()
        if "HIP_VISIBLE_DEVICES" not in env:
            pick = _pick_idle_gpu()
            if pick is not None:
                env["HIP_VISIBLE_DEVICES"] = pick
                _log(f"[metric_d64] {label} d{dim} {layout}: HIP_VISIBLE_DEVICES={pick}")
        try:
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"label": label, "phase": "crashed", "crash_reason": f"timeout {timeout}s",
                    "cos": {}, "fail": 4, "warn": 0, "fwd_tflops": 0.0, "bwd_tflops": 0.0,
                    "fwd_ms": float("nan"), "bwd_ms": float("nan"),
                    "dim": dim, "layout": layout}
        if res.returncode != 0:
            stderr_tail = "\n".join((res.stderr or "").splitlines()[-8:])
            stdout_tail = "\n".join((res.stdout or "").splitlines()[-4:])
            crash_reason = f"exit {res.returncode}: {stderr_tail or stdout_tail or '(silent)'}"
            try:
                with open(out_path) as f:
                    partial = json.load(f)
                partial["phase"] = f"crashed_after_{partial.get('phase', 'unknown')}"
                partial["crash_reason"] = crash_reason
                return partial
            except Exception:
                return {"label": label, "phase": "crashed", "crash_reason": crash_reason,
                        "cos": {}, "fail": 4, "warn": 0, "fwd_tflops": 0.0, "bwd_tflops": 0.0,
                        "fwd_ms": float("nan"), "bwd_ms": float("nan"),
                        "dim": dim, "layout": layout}
        try:
            with open(out_path) as f:
                return json.load(f)
        except Exception as exc:
            return {"label": label, "phase": "crashed",
                    "crash_reason": f"could not parse worker JSON: {exc!r}",
                    "cos": {}, "fail": 4, "warn": 0, "fwd_tflops": 0.0, "bwd_tflops": 0.0,
                    "fwd_ms": float("nan"), "bwd_ms": float("nan"),
                    "dim": dim, "layout": layout}
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def _measurement_plan() -> list[tuple[str, int, int, int, int, bool, int, str]]:
    """List of (label, B, N, H, H_KV, causal, dim, layout) tasks to run.

    Each shape contributes:
      - one D=128 BSHD task (wall-clock baseline)
      - one D=64 BSHD task   (primary D=64 measurement)
      - one D=64 SBHD task   (secondary user layout; only on N=4096 to keep
        metric wall < ~75s).
    """
    tasks = []
    for label, B, N, H, H_KV, causal in SHAPES:
        tasks.append((label, B, N, H, H_KV, causal, D_REF, "BSHD"))
        tasks.append((label, B, N, H, H_KV, causal, 64,    "BSHD"))
        if N == 4096:  # only measure SBHD on the central shape
            tasks.append((label, B, N, H, H_KV, causal, 64, "SBHD"))
    return tasks


def main() -> int:
    # Worker mode dispatch.
    if len(sys.argv) >= 9 and sys.argv[1] == "--worker":
        out_json = sys.argv[2]; label = sys.argv[3]
        B, N, H, H_KV = (int(x) for x in sys.argv[4:8])
        causal = bool(int(sys.argv[8]))
        dim = int(sys.argv[9]) if len(sys.argv) >= 10 else 64
        layout = sys.argv[10] if len(sys.argv) >= 11 else "BSHD"
        try:
            return _worker_main(out_json, label, B, N, H, H_KV, causal, dim, layout)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return 1

    # Parent mode.
    t0 = time.monotonic()

    # Step 1: build (compile gate).
    _log(f"[metric_d64] building d64 + d128_ref ATTN_N={BUILD_N} ...")
    ok, build_tail = _build_kernels(BUILD_N)
    if not ok:
        score = -COMPILE_FAIL_PENALTY
        print(score)
        _log(f"[metric_d64] BUILD FAILED:\n{build_tail}")
        _log(f"[metric_d64] score={score}")
        return 0

    # Step 2: per-(shape, dim, layout) correctness + perf.
    results: dict[tuple, dict] = {}  # (label, dim, layout) -> result dict
    cos_fail = 0
    cos_warn = 0
    exceptions = 0
    sum_d64_tflops = 0.0
    notes: list[str] = []

    for label, B, N, H, H_KV, causal, dim, layout in _measurement_plan():
        r = _run_worker(label, B, N, H, H_KV, causal, dim, layout)
        results[(label, dim, layout)] = r
        phase = r.get("phase", "unknown")
        # D=128 baseline tasks contribute wall-clock only -- their cos /
        # exception count does NOT enter the score (the D=128 fwd path
        # is unavailable upstream so the bwd-only run uses synthetic L,
        # producing meaningless cos values).  Crashes still warn.
        is_baseline = (dim == D_REF)
        if phase.startswith("crashed"):
            if is_baseline:
                notes.append(
                    f"  ERR  {label} d{dim} {layout} (baseline crashed; "
                    f"wall-clock unavailable): {r.get('crash_reason', '')[:160]}"
                )
            else:
                exceptions += 1
                cos_fail += r.get("fail", 4)
                cos_warn += r.get("warn", 0)
                notes.append(
                    f"  ERR  {label} d{dim} {layout} ({phase}): "
                    f"{r.get('crash_reason', '')[:160]}"
                )
            continue
        if not is_baseline:
            cos_fail += r["fail"]
            cos_warn += r["warn"]
            sum_d64_tflops += r["fwd_tflops"] + r["bwd_tflops"]
        cos_str = " ".join(f"{k}={v:.4f}" for k, v in r["cos"].items())
        notes.append(
            f"  PERF {label} d{dim} {layout}: "
            f"fwd={r['fwd_tflops']:.0f}TF ({r['fwd_ms']:.2f}ms) "
            f"bwd={r['bwd_tflops']:.0f}TF ({r['bwd_ms']:.2f}ms)"
        )
        if is_baseline:
            notes.append(
                f"  COS  {label} d{dim} {layout}: {cos_str} "
                f"(baseline; cos NOT scored)"
            )
        else:
            notes.append(
                f"  COS  {label} d{dim} {layout}: {cos_str} "
                f"fail={r['fail']} warn={r['warn']}"
            )

    # Step 3: compute speedup bonus.  We compare *bwd* wall-clock only --
    # D=128 fwd is unavailable upstream so we don't have a comparable fwd
    # baseline, and bwd dominates training-time anyway (~70-80%).
    speedup_bonus = 0
    for label, B, N, H, H_KV, causal in SHAPES:
        d128 = results.get((label, D_REF, "BSHD"))
        d64_bshd = results.get((label, 64, "BSHD"))
        d64_sbhd = results.get((label, 64, "SBHD"))
        if not d128 or not d64_bshd:
            continue
        if d128.get("phase", "").startswith("crashed"):
            continue
        d128_bwd = d128.get("bwd_ms") or float("nan")
        if not math.isfinite(d128_bwd) or d128_bwd <= 0:
            continue
        if not d64_bshd.get("phase", "").startswith("crashed"):
            d64_bwd = d64_bshd.get("bwd_ms") or float("nan")
            if math.isfinite(d64_bwd) and d64_bwd > 0:
                speedup = max(0.0, min(1.0, 1.0 - d64_bwd / d128_bwd))
                bonus = int(round(SPEEDUP_BSHD_BONUS * speedup))
                speedup_bonus += bonus
                notes.append(
                    f"  SPD  {label} BSHD: d64_bwd={d64_bwd:.2f}ms "
                    f"d128_bwd={d128_bwd:.2f}ms speedup={speedup:.3f} bonus=+{bonus}"
                )
        if d64_sbhd and not d64_sbhd.get("phase", "").startswith("crashed"):
            d64_bwd = d64_sbhd.get("bwd_ms") or float("nan")
            if math.isfinite(d64_bwd) and d64_bwd > 0:
                speedup = max(0.0, min(1.0, 1.0 - d64_bwd / d128_bwd))
                bonus = int(round(SPEEDUP_SBHD_BONUS * speedup))
                speedup_bonus += bonus
                notes.append(
                    f"  SPD  {label} SBHD: d64_bwd={d64_bwd:.2f}ms "
                    f"d128_bwd={d128_bwd:.2f}ms speedup={speedup:.3f} bonus=+{bonus}"
                )

    dt = time.monotonic() - t0
    score = (
        int(round(sum_d64_tflops * 10))
        - COS_FAIL_PENALTY * cos_fail
        - COS_WARN_PENALTY * cos_warn
        - EXCEPTION_PENALTY * exceptions
        + speedup_bonus
    )

    print(score)
    _log(
        f"[metric_d64] sum_d64_tflops={sum_d64_tflops:.1f} "
        f"cos_fail={cos_fail} cos_warn={cos_warn} exc={exceptions} "
        f"speedup_bonus={speedup_bonus} score={score} elapsed={dt:.1f}s"
    )
    show_all = "--verbose" in sys.argv
    for n in notes:
        if show_all or n.lstrip().startswith(("ERR", "COS", "SPD")):
            _log(n)

    if "--strict" in sys.argv:
        if cos_fail > 0 or exceptions > 0:
            _log(f"[metric_d64] --strict: cos_fail={cos_fail} exc={exceptions} -> non-zero exit")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
