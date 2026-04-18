#!/bin/bash
# R38 Dev D — nm-based dead-code gate (drop-in replacement for md5
# build hygiene).
#
# R37 Dev C documented that md5 is unreliable in this env: two consecutive
# identical hipcc compilations of the same source produce different md5
# hashes (likely embedded build IDs / timestamps). The nm-gate verifies
# build hygiene structurally instead: a compile-flag-gated feature is
# "dead" in the default build iff `nm -D <so> | grep <symbol>` returns 0
# matches. If the default build contains 0 instances of a feature's
# symbol pattern, the feature is provably not invoked at runtime.
#
# Usage:
#   ./r38_nm_gate.sh <so_path>                  # full catalog (all features)
#   ./r38_nm_gate.sh <so_path> hbshrink hbn     # subset
#   ./r38_nm_gate.sh --json <so_path>           # JSON output
#
# Exit code:
#   0 — every catalog feature has 0 symbols (default build hygiene PASS)
#   1 — at least one cataloged "default-off" feature has >0 symbols (FAIL)
#   2 — usage error
#
# When invoked on a NON-default build (e.g. PROD .so built with
# -DMXFP8_CRR_BLK_M=128 -DMXFP8_CRR_HBSHRINK_PIPELINE=1), the gate is NOT
# expected to return 0 for the active feature — it instead reports
# whether the EXPECTED feature symbols are PRESENT (so the prod .so
# actually contains the feature). Use --expect-active <feature> to flip
# the assertion for that one feature.
#
# Catalog (R38 NEW baseline — established on main HEAD default build):
#
# | feature       | symbol pattern                   | default count | source                                     |
# |---------------|----------------------------------|---------------|--------------------------------------------|
# | hbshrink (B1) | hbshrink                         | 0             | crr_mxfp8_exact_8wave_hbshrink_fastpath.inc|
# | hbn (R38 A/B) | hbn| HBN_                        | 0             | (placeholder; HBN variant not yet landed)  |
# | subrbm        | subrbm                           | 0             | crr_mxfp8_exact_8wave_subrbm_fastpath.inc  |
# | warpsm4       | warpsm4                          | 0             | crr_mxfp8_exact_8wave_warpsm4_fastpath.inc |
# | double_pump   | double_pump                      | 0             | crr_exact_8wave_double_pump_fastpath.inc   |
# | mxfp8_4wave   | _4wave                           | 0             | *_mxfp8_4wave_fastpath.inc                 |
# | rect          | _8wave_rect                      | 0             | *_mxfp8_exact_8wave_rect_fastpath.inc      |
#
# Note (R38 Dev D): V2-RCR predicates (R36 Dev C) and V2-RRR predicates
# (R36 Dev B) are runtime shape-gated, NOT compile-flag-gated. Their kernel
# symbols (rcr_exact_8wave_scaled_kernel, rrr_exact_8wave_scaled_kernel,
# dispatch_rcr_exact_8wave_scaled_v2, dispatch_rrr_exact_8wave_scaled_v2)
# are ALWAYS present in every build. The nm-gate verifies the inverse for
# them: presence (count >= 1) rather than absence. Use --check-present
# rcr_v2 / rrr_v2 to flip the assertion.

set -u

JSON_OUT=0
EXPECT_ACTIVE=()
CHECK_PRESENT=()
SO_PATH=""
USER_FEATURES=()

while [ $# -gt 0 ]; do
  case "$1" in
    --json) JSON_OUT=1; shift ;;
    --expect-active) EXPECT_ACTIVE+=("$2"); shift 2 ;;
    --check-present) CHECK_PRESENT+=("$2"); shift 2 ;;
    -h|--help)
      sed -n '1,50p' "$0"
      exit 0
      ;;
    *)
      if [ -z "$SO_PATH" ]; then
        SO_PATH="$1"
      else
        USER_FEATURES+=("$1")
      fi
      shift
      ;;
  esac
done

if [ -z "$SO_PATH" ]; then
  echo "Usage: $0 [--json] [--expect-active <feature>] [--check-present <feature>] <so_path> [feature...]" >&2
  exit 2
fi
if [ ! -f "$SO_PATH" ]; then
  echo "ERROR: .so not found: $SO_PATH" >&2
  exit 2
fi

# Catalog: feature_name -> grep pattern
declare -A FEATURES
FEATURES[hbshrink]="hbshrink"
FEATURES[hbn]="\\bhbn\\b\\|HBN_"
FEATURES[subrbm]="subrbm"
FEATURES[warpsm4]="warpsm4"
FEATURES[double_pump]="double_pump"
FEATURES[mxfp8_4wave]="_4wave"
FEATURES[rect]="_8wave_rect"
# Runtime-gated (always present — checked with --check-present):
FEATURES[rcr_v2]="dispatch_rcr_exact_8wave_scaled_v2"
FEATURES[rrr_v2]="dispatch_rrr_exact_8wave_scaled_v2"
FEATURES[crr_v2]="dispatch_crr_exact_8wave_scaled_v2"

# Default catalog order (deterministic output)
DEFAULT_ORDER=(hbshrink hbn subrbm warpsm4 double_pump mxfp8_4wave rect rcr_v2 rrr_v2 crr_v2)

# Decide which features to check
if [ ${#USER_FEATURES[@]} -gt 0 ]; then
  FEATURES_TO_CHECK=("${USER_FEATURES[@]}")
else
  FEATURES_TO_CHECK=("${DEFAULT_ORDER[@]}")
fi

# Symbol counts (full and demangled)
NM_RAW=$(nm -D "$SO_PATH" 2>&1)
NM_DEMANGLED=$(nm -D -C "$SO_PATH" 2>&1)

is_in_list() {
  local needle=$1; shift
  for x in "$@"; do
    [ "$x" = "$needle" ] && return 0
  done
  return 1
}

OVERALL_FAIL=0

if [ $JSON_OUT -eq 1 ]; then
  echo '{"so":"'$SO_PATH'","features":['
fi

first=1
for feat in "${FEATURES_TO_CHECK[@]}"; do
  pat="${FEATURES[$feat]:-}"
  if [ -z "$pat" ]; then
    echo "WARN: unknown feature '$feat' (no catalog entry); skipping" >&2
    continue
  fi
  count=$(echo "$NM_DEMANGLED" | grep -c "$pat" || true)

  # Decide expectation
  expected="absent"
  if is_in_list "$feat" "${EXPECT_ACTIVE[@]}"; then expected="present"; fi
  if is_in_list "$feat" "${CHECK_PRESENT[@]}"; then expected="present"; fi
  case "$feat" in
    rcr_v2|rrr_v2|crr_v2) expected="present" ;;
  esac

  result="PASS"
  if [ "$expected" = "absent" ] && [ "$count" -gt 0 ]; then
    result="FAIL_unexpected_present"
    OVERALL_FAIL=1
  elif [ "$expected" = "present" ] && [ "$count" -eq 0 ]; then
    result="FAIL_missing"
    OVERALL_FAIL=1
  fi

  if [ $JSON_OUT -eq 1 ]; then
    if [ $first -eq 0 ]; then echo ","; fi
    first=0
    # Escape backslashes for JSON
    pat_json=$(echo "$pat" | sed 's/\\/\\\\/g')
    printf '  {"feature":"%s","pattern":"%s","count":%d,"expected":"%s","result":"%s"}' \
      "$feat" "$pat_json" "$count" "$expected" "$result"
  else
    printf '  %-14s pattern=%-44s count=%-4d expected=%-8s result=%s\n' \
      "$feat" "$pat" "$count" "$expected" "$result"
  fi
done

if [ $JSON_OUT -eq 1 ]; then
  echo ''
  echo '],'
  echo '"overall":'$([ $OVERALL_FAIL -eq 0 ] && echo '"PASS"' || echo '"FAIL"')
  echo '}'
else
  echo "OVERALL: $([ $OVERALL_FAIL -eq 0 ] && echo PASS || echo FAIL)"
fi

exit $OVERALL_FAIL
