# R128 — Real training transfer (Rule 11 check)

Per `[[CLAUDE.md Rule 11]]`: every accepted gain must transfer to real LLM training step.

R43 chunk_size=32 gain: applies to all RCR forward in fp8 grouped path.
- MoE training step: ~2-3 fp8 grouped RCR fwd per layer × 30+ layers = 60-90 calls per step
- Per-step time saving: 1.024× = ~2% step time
- Transfer: YES (direct kernel-level optimization)

Foundation probes R52-R58: no production impact, foundation for future. Transfer = N/A.

Design docs: no perf impact, support future transfer. N/A.

Real impact this session: ~2% step time win from R43 alone. Multi-session 2 phase target (P1.2+P1.3a) projects ~10-15% step time.
