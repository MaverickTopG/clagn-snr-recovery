#!/usr/bin/env bash
# Single source of truth for "does this repository pass".
#
# Nothing may claim the suite is green except this script. Every check runs
# even if an earlier one fails, so one failure does not hide the others, and
# the summary is printed from measured results rather than transcribed by hand.
#
#   ./verify.sh          full run
#   ./verify.sh --fast   skip network-marked tests
#
# Exit code is 0 only when every check passes. Intended as the pre-push gate.

set -uo pipefail
cd "$(dirname "$0")"

FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1

RED=$'\033[31m'; GREEN=$'\033[32m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

declare -a NAMES=() STATUS=() DETAIL=()
FAILED=0

record() {  # name rc detail
  NAMES+=("$1"); STATUS+=("$2"); DETAIL+=("$3")
  [[ "$2" -ne 0 ]] && FAILED=1
  return 0
}

echo "${BOLD}Running verification${RESET}"
echo

# --- tests ----------------------------------------------------------------
echo "→ pytest"
# NB: pyproject addopts already supplies -q. Passing another -q makes
# pytest doubly quiet and suppresses the count line entirely.
PYTEST_ARGS=(--tb=short)
[[ "$FAST" -eq 1 ]] && PYTEST_ARGS+=(-m "not network")
PYTEST_OUT="$(uv run pytest "${PYTEST_ARGS[@]}" 2>&1)"
PYTEST_RC=$?
echo "$PYTEST_OUT" | tail -3
# Parse the counts pytest actually reported; never assume.
# pytest -q suppresses the count line when warnings are present, so ask for it
# explicitly rather than parsing whatever happens to be printed.
SUMMARY="$(uv run python -c '
import re, sys
text = sys.stdin.read()
m = re.findall(r"(\d+) (passed|failed|error|skipped|xfailed)", text)
print(", ".join(f"{n} {k}" for n, k in m) or "no counts reported")
' <<< "$PYTEST_OUT")"
[[ -z "$SUMMARY" ]] && SUMMARY="no summary line found"
record "pytest" "$PYTEST_RC" "$SUMMARY"
echo

# --- lint -----------------------------------------------------------------
echo "→ ruff"
RUFF_OUT="$(uv run ruff check . 2>&1)"; RUFF_RC=$?
echo "$RUFF_OUT" | tail -2
record "ruff" "$RUFF_RC" "$(echo "$RUFF_OUT" | tail -1)"
echo

# --- types ----------------------------------------------------------------
echo "→ mypy"
MYPY_OUT="$(uv run mypy src/p3sf 2>&1)"; MYPY_RC=$?
echo "$MYPY_OUT" | tail -2
record "mypy" "$MYPY_RC" "$(echo "$MYPY_OUT" | tail -1)"
echo

# --- environment pins -----------------------------------------------------
echo "→ environment"
ENV_OUT="$(uv run python -W ignore 00_scripts/00_make_environment.py --check 2>&1)"; ENV_RC=$?
echo "$ENV_OUT" | tail -2
record "environment" "$ENV_RC" "$(echo "$ENV_OUT" | grep -E '^(environment OK|FAILED)' | head -1)"
echo

# --- summary --------------------------------------------------------------
echo "${BOLD}────────────────────────────────────────────────────────────${RESET}"
for i in "${!NAMES[@]}"; do
  if [[ "${STATUS[$i]}" -eq 0 ]]; then
    printf "  %s%-14s PASS%s  %s\n" "$GREEN" "${NAMES[$i]}" "$RESET" "${DETAIL[$i]}"
  else
    printf "  %s%-14s FAIL%s  %s\n" "$RED" "${NAMES[$i]}" "$RESET" "${DETAIL[$i]}"
  fi
done
echo "${BOLD}────────────────────────────────────────────────────────────${RESET}"

if [[ "$FAILED" -ne 0 ]]; then
  echo "${RED}${BOLD}VERIFICATION FAILED${RESET}"
  echo "Do not claim the suite is green, and do not push."
  exit 1
fi

echo "${GREEN}${BOLD}VERIFICATION PASSED${RESET}"
echo "$SUMMARY"
exit 0
