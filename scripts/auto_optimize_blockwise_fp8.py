#!/usr/bin/env python3
"""Auto-optimize daemon for HipKittens blockwise FP8 GEMM kernel.

Architecture (simplified port of Primus-Turbo's auto_optimize_gpt_oss_fp8.py
for our LOCAL single-repo MI300X setup — no docker, no remote, no rsync):

  this checkout (REPO auto-derived from __file__, branch feat/fp8-blockwise-mi300x)
     ├── kernels/gemm/fp8fp32/mi300x/blockwise_8192/             ← claude edits here
     ├── scripts/_metric_blockwise_fp8_{target,loser}_shapes.py   ← daemon runs this
     └── auto_optimize_logs/<run>/round_NNN/{prompt.md,claude.{log,jsonl}}

Per round:
  1. build_prompt          — task .md + history + recent git log + state
  2. spawn `claude --print` — fed prompt via stdin, edits files + git commit,
                              exits when done. 8-min stall watchdog,
                              45-min hard timeout.
  3. run metric            — daemon's own canonical run; the score that counts
  4. update state          — best/streak/early-stop, write summary.json

Usage:
  python3 scripts/auto_optimize_blockwise_fp8.py \\
      --rounds 200 --patience 50

  # Resume from prior run:
  python3 scripts/auto_optimize_blockwise_fp8.py \\
      --resume-state auto_optimize_logs/<run>/summary.json \\
      --rounds 200 --patience 50

Typically launched via scripts/launch_auto_optimize_blockwise_fp8.sh which
wraps this in nohup + setsid for true background detachment.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Pinned constants

REPO          = Path(__file__).resolve().parents[1]
# Round-100+ pivot: the full 54-shape metric is ~92% saturated. Most of the
# remaining headroom sits in a handful of LOSER pairs (HK below Triton). The
# daemon now defaults to the loser-only metric + loser-only task so each
# round's edits land where they matter. To restore the full-target run:
#   --task-file scripts/_task_blockwise_fp8.md
#   --metric-cmd 'python3 scripts/_metric_blockwise_fp8_target_shapes.py'
DEFAULT_TASK   = "scripts/_task_blockwise_fp8_losers_extra.md"
DEFAULT_METRIC = "python3 scripts/_metric_blockwise_fp8_loser_shapes.py"
DEFAULT_METRIC_NAME = "blockwise_fp8_loser_pairs_score"
DEFAULT_MODEL = "claude-opus-4-7[1m]"
DEFAULT_EFFORT = "max"


# ─────────────────────────────────────────────────────────────────────────────
# State

@dataclass
class RoundResult:
    index: int
    started_at: str
    finished_at: str
    duration_s: float
    metric: Optional[float]
    best_so_far: Optional[float]
    improved: bool
    head_sha_before: str
    head_sha_after: str
    claude_exit_code: int
    log_dir: str

    def as_dict(self) -> dict:
        return {
            "index": self.index, "started_at": self.started_at,
            "finished_at": self.finished_at, "duration_s": round(self.duration_s, 2),
            "metric": self.metric, "best_so_far": self.best_so_far,
            "improved": self.improved, "head_sha_before": self.head_sha_before,
            "head_sha_after": self.head_sha_after,
            "claude_exit_code": self.claude_exit_code, "log_dir": self.log_dir,
        }


@dataclass
class TrajectoryState:
    rounds: list[RoundResult] = field(default_factory=list)
    best_metric: Optional[float] = None
    best_sha: Optional[str] = None
    rounds_without_improvement: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Utilities

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def banner(msg: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def section(msg: str) -> None:
    print(f"\n--- {msg} ---", flush=True)


def get_head_sha(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=cwd, text=True).strip()
    except Exception:
        return ""


def get_recent_log(cwd: Path, n: int = 8) -> str:
    try:
        return subprocess.check_output(
            ["git", "log", f"-{n}", "--oneline"], cwd=cwd, text=True).strip()
    except Exception:
        return ""


def get_short_status(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], cwd=cwd, text=True).strip()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Args

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--rounds", type=int, default=200,
                   help="max rounds (default 200)")
    p.add_argument("--patience", type=int, default=50,
                   help="early-stop after N consecutive no-improvement rounds")
    p.add_argument("--min-delta", type=float, default=0.0,
                   help="metric improvement needed to reset streak (default 0)")
    p.add_argument("--task-file", type=str, default=DEFAULT_TASK,
                   help="path (relative to REPO) of the task .md")
    p.add_argument("--metric-cmd", type=str, default=DEFAULT_METRIC,
                   help="shell command run from REPO to score one config")
    p.add_argument("--metric-name", type=str, default=DEFAULT_METRIC_NAME)
    p.add_argument("--metric-timeout", type=int, default=60 * 8,
                   help="seconds to wait for one metric run")
    p.add_argument("--metric-samples", type=int, default=1,
                   help="re-run metric N times per round, take median")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--effort", type=str, default=DEFAULT_EFFORT,
                   choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--round-timeout", type=int, default=60 * 45,
                   help="hard timeout for claude per round (default 45 min)")
    p.add_argument("--stall-timeout", type=int, default=60 * 8,
                   help="kill claude after N seconds with no stdout (default 8 min)")
    p.add_argument("--log-dir", type=str, default="",
                   help="absolute log directory (default auto-timestamped)")
    p.add_argument("--prompt-extra", type=str, default="")
    p.add_argument("--prompt-extra-file", type=str, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="skip claude, exercise metric+state only")
    p.add_argument("--resume-state", type=str, default="",
                   help="resume from a prior summary.json")
    p.add_argument("--skip-baseline", action="store_true",
                   help="skip the initial baseline metric run")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Metric

def _run_metric_once(metric_cmd: str, timeout: int) -> Optional[float]:
    cmd = shlex.split(metric_cmd)
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[metric] TIMEOUT after {timeout}s", flush=True)
        return None
    if r.returncode != 0:
        print(f"[metric] returncode {r.returncode}", flush=True)
        print(f"[metric] stderr tail:\n{r.stderr[-1500:]}", flush=True)
        return None
    last_line = ""
    for line in reversed(r.stdout.splitlines()):
        if line.strip():
            last_line = line.strip()
            break
    try:
        return float(last_line)
    except (ValueError, TypeError):
        print(f"[metric] last stdout line not float: {last_line!r}", flush=True)
        print(f"[metric] stdout tail:\n{r.stdout[-1500:]}", flush=True)
        return None


def run_metric(metric_cmd: str, timeout: int, samples: int = 1) -> Optional[float]:
    section(f"metric: {metric_cmd}  (samples={samples})")
    vals: list[float] = []
    for i in range(samples):
        v = _run_metric_once(metric_cmd, timeout)
        print(f"[metric] sample {i+1}/{samples}: {v}", flush=True)
        if v is None:
            return None
        vals.append(v)
    vals.sort()
    median = vals[len(vals) // 2]
    print(f"[metric] median = {median}  (samples = {vals})", flush=True)
    return median


# ─────────────────────────────────────────────────────────────────────────────
# Prompt

def build_prompt(args: argparse.Namespace, state: TrajectoryState,
                 round_idx: int, baseline_metric: Optional[float],
                 head_sha: str, recent_log: str, short_status: str) -> str:
    last = state.rounds[-1] if state.rounds else None
    last_metric = last.metric if last else baseline_metric
    last_improved = "yes" if (last and last.improved) else "no"

    history_lines = []
    for r in state.rounds[-5:]:
        history_lines.append(
            f"  - round {r.index}: metric={r.metric}, best={r.best_so_far}, "
            f"improved={r.improved}, sha {r.head_sha_before[:8]}->{r.head_sha_after[:8]}, "
            f"log={r.log_dir}"
        )
    history_block = "\n".join(history_lines) if history_lines else "  (none yet)"

    task_path = REPO / args.task_file
    task_text = task_path.read_text(encoding="utf-8") if task_path.exists() \
                else f"(task file missing: {task_path})"

    return f"""[autonomous round {round_idx} / {args.rounds}, scheduled by daemon]

# Live state from the daemon (read this BEFORE the embedded task spec below)

- **Repo**: {REPO} (branch `feat/fp8-blockwise-mi300x`)
- **Metric ({args.metric_name})**: higher is better
  - baseline (before any round): {baseline_metric}
  - best so far: {state.best_metric} (sha {state.best_sha[:8] if state.best_sha else "n/a"})
  - last round: {last_metric} (improved={last_improved})
  - consecutive rounds without improvement: {state.rounds_without_improvement} / {args.patience}
- **Recent rounds**:
{history_block}
- **Recent git log** (REPO HEAD = {head_sha[:8]}):
```
{recent_log or "(empty)"}
```
- **Working tree status** (should be clean — daemon expects no uncommitted state at round start):
```
{short_status or "(clean)"}
```

The daemon will run `{args.metric_cmd}` from `{REPO}` after you exit and use that
as the canonical score for this round. Anything you do yourself with the metric is
debug-only.

---

# Embedded task spec (from `{args.task_file}`)

{task_text}

{args.prompt_extra}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Claude invocation

def _format_tool_call_summary(ev: dict) -> Optional[str]:
    if ev.get("type") != "assistant":
        return None
    msg = ev.get("message") or {}
    summaries = []
    for blk in msg.get("content") or []:
        if blk.get("type") == "tool_use":
            name = blk.get("name", "?")
            inp = blk.get("input") or {}
            short = ""
            if name == "Bash":
                short = (inp.get("command") or "")[:120]
            elif name in ("Read", "Edit", "Write"):
                short = inp.get("file_path", "")
            elif name == "Grep":
                short = inp.get("pattern", "")
            summaries.append(f"  [tool] {name}({short})")
    return "\n".join(summaries) if summaries else None


def run_claude_round(args: argparse.Namespace, prompt: str,
                     log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "prompt.md").write_text(prompt)

    cmd = [
        "claude", "--print",
        "--model", args.model,
        "--effort", args.effort,
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--permission-mode", "acceptEdits",
        "--add-dir", str(REPO),
    ]
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"[claude] {pretty}  (prompt via stdin, {len(prompt)} chars)", flush=True)

    log_path = log_dir / "claude.log"
    raw_path = log_dir / "claude.jsonl"

    def _handle_line(line: str, logf, rawf) -> None:
        rawf.write(line); rawf.flush()
        s = line.strip()
        if not s:
            return
        try:
            ev = json.loads(s)
        except json.JSONDecodeError:
            logf.write(line); logf.flush()
            sys.stdout.write(line); sys.stdout.flush()
            return
        ev_type = ev.get("type")
        if ev_type == "assistant":
            msg = ev.get("message") or {}
            for blk in msg.get("content") or []:
                if blk.get("type") == "text":
                    txt = blk.get("text", "")
                    logf.write(txt + "\n")
                    sys.stdout.write(txt + "\n"); sys.stdout.flush()
            summary = _format_tool_call_summary(ev)
            if summary:
                logf.write(summary + "\n")
                sys.stdout.write(summary + "\n"); sys.stdout.flush()
        elif ev_type == "result":
            dur = ev.get("duration_ms", 0)
            cost = ev.get("total_cost_usd")
            msg = (f"  [claude] result: duration={dur}ms "
                   f"is_error={ev.get('is_error', False)} cost_usd={cost}")
            logf.write(msg + "\n")
            sys.stdout.write(msg + "\n"); sys.stdout.flush()
        elif ev_type == "system":
            logf.write(line); logf.flush()
        logf.flush()

    def _terminate(proc: subprocess.Popen, reason: str, logf) -> int:
        msg = f"\n[claude] {reason}; sending SIGTERM"
        print(msg, flush=True); logf.write(msg + "\n")
        try: proc.send_signal(signal.SIGTERM)
        except Exception: pass
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try: proc.send_signal(signal.SIGKILL)
            except Exception: pass
            try: proc.wait(timeout=5)
            except Exception: pass
        return proc.returncode if proc.returncode is not None else 137

    proc = subprocess.Popen(
        cmd, cwd=REPO, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdin is not None
    proc.stdin.write(prompt)
    proc.stdin.close()

    last_output = time.monotonic()
    started_at = last_output

    with open(log_path, "w") as logf, open(raw_path, "w") as rawf:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                last_output = time.monotonic()
                _handle_line(line, logf, rawf)
                # Watchdogs
                if time.monotonic() - started_at > args.round_timeout:
                    return _terminate(proc, f"round timeout {args.round_timeout}s exceeded", logf)
                if time.monotonic() - last_output > args.stall_timeout:
                    return _terminate(proc, f"stall timeout {args.stall_timeout}s exceeded", logf)
        except KeyboardInterrupt:
            return _terminate(proc, "KeyboardInterrupt", logf)

    rc = proc.wait()
    print(f"[claude] exited rc={rc}", flush=True)
    return rc


# ─────────────────────────────────────────────────────────────────────────────
# Summary

def write_summary(summary_path: Path, args: argparse.Namespace,
                  state: TrajectoryState, baseline: Optional[float]) -> None:
    summary = {
        "started_at": getattr(write_summary, "_start", now_iso()),
        "metric_name": args.metric_name,
        "metric_cmd": args.metric_cmd,
        "model": args.model,
        "effort": args.effort,
        "rounds_planned": args.rounds,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "baseline_metric": baseline,
        "best_metric": state.best_metric,
        "best_sha": state.best_sha,
        "rounds_run": len(state.rounds),
        "rounds_without_improvement": state.rounds_without_improvement,
        "rounds": [r.as_dict() for r in state.rounds],
    }
    summary_path.write_text(json.dumps(summary, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main loop

def main() -> int:
    args = parse_args()
    if args.prompt_extra_file:
        try:
            args.prompt_extra = Path(args.prompt_extra_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"--prompt-extra-file unreadable: {exc}", file=sys.stderr)
            return 2

    if not REPO.is_dir():
        print(f"REPO missing: {REPO}", file=sys.stderr); return 2
    if not (REPO / args.task_file).is_file():
        print(f"task file missing: {REPO / args.task_file}", file=sys.stderr); return 2
    if shutil.which("claude") is None and not args.dry_run:
        print("`claude` CLI not on PATH", file=sys.stderr); return 2

    log_dir = (Path(args.log_dir) if args.log_dir else
               REPO / "auto_optimize_logs" /
               f"blockwise_fp8_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.json"
    write_summary._start = now_iso()  # type: ignore[attr-defined]

    state = TrajectoryState()
    baseline: Optional[float] = None
    round_offset = 0

    if args.resume_state:
        try:
            saved = json.loads(Path(args.resume_state).read_text())
        except Exception as exc:
            print(f"--resume-state unreadable: {exc}", file=sys.stderr); return 2
        baseline = saved.get("baseline_metric")
        state.best_metric = saved.get("best_metric")
        state.best_sha = saved.get("best_sha")
        for rd in saved.get("rounds") or []:
            state.rounds.append(RoundResult(
                index=rd["index"], started_at=rd["started_at"],
                finished_at=rd["finished_at"], duration_s=rd["duration_s"],
                metric=rd.get("metric"), best_so_far=rd.get("best_so_far"),
                improved=rd.get("improved", False),
                head_sha_before=rd.get("head_sha_before", ""),
                head_sha_after=rd.get("head_sha_after", ""),
                claude_exit_code=rd.get("claude_exit_code", 0),
                log_dir=rd.get("log_dir", ""),
            ))
        round_offset = saved.get("rounds_run", len(state.rounds))
        streak = 0
        for r in reversed(state.rounds):
            if r.improved:
                break
            streak += 1
        state.rounds_without_improvement = streak
        write_summary(summary_path, args, state, baseline)

    try:
        if args.resume_state:
            remaining = max(args.rounds - round_offset, 0)
            banner(f"AUTO-OPTIMIZE RESUME | restored {round_offset} | "
                   f"--rounds={args.rounds} ({remaining} more) | "
                   f"streak={state.rounds_without_improvement} | "
                   f"baseline={baseline} | best={state.best_metric} | log={log_dir}")
            if remaining <= 0:
                banner("nothing to do.")
                return 0
            range_iter = range(round_offset + 1, args.rounds + 1)
        else:
            banner(f"AUTO-OPTIMIZE start | rounds={args.rounds} | "
                   f"patience={args.patience} | log={log_dir}")
            if not args.skip_baseline:
                section("baseline metric")
                baseline = run_metric(args.metric_cmd, args.metric_timeout,
                                      samples=args.metric_samples)
                state.best_metric = baseline
                state.best_sha = get_head_sha(REPO)
            write_summary(summary_path, args, state, baseline)
            range_iter = range(1, args.rounds + 1)

        for i in range_iter:
            banner(f"ROUND {i}/{args.rounds} | best={state.best_metric} | "
                   f"streak={state.rounds_without_improvement}/{args.patience}")
            sha_before = get_head_sha(REPO)
            recent_log = get_recent_log(REPO)
            short_status = get_short_status(REPO)
            prompt = build_prompt(args, state, i, baseline, sha_before,
                                  recent_log, short_status)

            round_dir = log_dir / f"round_{i:03d}"
            started_at = now_iso()
            t0 = time.monotonic()
            if args.dry_run:
                print("[dry-run] skipping claude", flush=True)
                round_dir.mkdir(parents=True, exist_ok=True)
                (round_dir / "prompt.md").write_text(prompt)
                claude_exit = 0
            else:
                claude_exit = run_claude_round(args, prompt, round_dir)
            duration = time.monotonic() - t0

            sha_after = get_head_sha(REPO)
            metric = run_metric(args.metric_cmd, args.metric_timeout,
                                samples=args.metric_samples)

            improved = False
            if metric is not None:
                if state.best_metric is None or metric > (state.best_metric + args.min_delta):
                    state.best_metric = metric
                    state.best_sha = sha_after
                    state.rounds_without_improvement = 0
                    improved = True
                else:
                    state.rounds_without_improvement += 1
            else:
                state.rounds_without_improvement += 1

            result = RoundResult(
                index=i, started_at=started_at, finished_at=now_iso(),
                duration_s=duration, metric=metric, best_so_far=state.best_metric,
                improved=improved, head_sha_before=sha_before, head_sha_after=sha_after,
                claude_exit_code=claude_exit,
                log_dir=str(round_dir.relative_to(log_dir.parent))
                        if round_dir.exists() else "",
            )
            state.rounds.append(result)
            write_summary(summary_path, args, state, baseline)

            print(f"[round {i}] metric={metric} best={state.best_metric} "
                  f"improved={improved} streak={state.rounds_without_improvement}/"
                  f"{args.patience} duration={duration:.1f}s", flush=True)

            if state.rounds_without_improvement >= args.patience:
                banner(f"EARLY-STOP: no improvement for {args.patience} rounds. "
                       f"Best={state.best_metric} at SHA {state.best_sha}.")
                break

        banner(f"AUTO-OPTIMIZE done | rounds_run={len(state.rounds)} | "
               f"baseline={baseline} | best={state.best_metric} | "
               f"best_sha={state.best_sha}")
        return 0
    except KeyboardInterrupt:
        banner("interrupted by user (Ctrl+C)")
        return 130
    finally:
        write_summary(summary_path, args, state, baseline)


if __name__ == "__main__":
    raise SystemExit(main())
