#!/usr/bin/env python3
"""R16A 5-run verify — SKIPPED.

All 5 R16A compound (iterilp + regclassglob) candidates FAILED the
single-shot smoke gate (>=+0.5pp on top of iterilp-winner). Smoke
deltas were ALL NEGATIVE (range -0.32pp to -14.82pp). Per the R16A
methodology, no candidate qualifies for 5-run verification.

This file exists only as a registered artifact placeholder. The
negative finding is recorded in bench_round16_optA_smoke.json and
round16_optA_verdict.md.
"""
import json, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    out = {
        "status": "SKIPPED",
        "reason": "0/5 candidates passed smoke gate (>=+0.5pp vs iterilp-winner)",
        "smoke_deltas_pp_vs_iterilp": {
            "S1": -14.82, "S2": -1.33, "S3": -0.43, "S4": -0.32, "S5": -1.20
        },
        "see": ["bench_round16_optA_smoke.json", "round16_optA_verdict.md"],
    }
    with open(os.path.join(SCRIPT_DIR, "bench_round16_optA_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
