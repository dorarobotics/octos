#!/usr/bin/env bash
# Local CI: mirrors GitHub Actions (.github/workflows/ci.yml) + focused subsystem tests.
# Usage: ./scripts/ci.sh [--quick] [--subsystem <name>]
#
# Subsystems: core, llm, agent, pipeline, bus, cli, memory
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QUICK=false
SUBSYSTEM=""
for arg in "$@"; do
    case "$arg" in
        --quick)    QUICK=true ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--subsystem <name>]"
            echo "Subsystems: core, llm, agent, pipeline, bus, cli, memory"
            exit 0
            ;;
        --subsystem) shift; SUBSYSTEM="${1:-}" ;;
        core|llm|agent|pipeline|bus|cli|memory)
            SUBSYSTEM="$arg" ;;
    esac
done

PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }
section() { echo ""; echo "--- $1 ---"; }

# ── 1. Format (mirrors CI: cargo fmt --all -- --check) ───────────────
section "Format Check"
if cargo fmt --all -- --check 2>/dev/null; then
    pass "cargo fmt"
else
    fail "cargo fmt (run: cargo fmt --all)"
fi

# ── 2. Clippy (mirrors CI: -D warnings) ──────────────────────────────
section "Clippy Lint"
if cargo clippy --workspace -- -D warnings 2>&1 | tail -3; then
    pass "cargo clippy"
else
    fail "cargo clippy"
fi

# ── 3. Tests ──────────────────────────────────────────────────────────
if [ -n "$SUBSYSTEM" ]; then
    # Focused subsystem test
    section "Subsystem Tests: $SUBSYSTEM"
    CRATE="crew-$SUBSYSTEM"
    if cargo test -p "$CRATE" 2>&1 | tee /tmp/crew-ci-sub.log | tail -5; then
        SUB_PASS=$(grep "^test result:" /tmp/crew-ci-sub.log | awk '{for(i=1;i<=NF;i++){if($i~/passed/){gsub(/[^0-9]/,"",$i);p+=$i}}}END{print p+0}')
        pass "$CRATE tests ($SUB_PASS passed)"
    else
        fail "$CRATE tests"
    fi
else
    # Full workspace test (mirrors CI: cargo test --workspace)
    section "Workspace Tests"
    if cargo test --workspace 2>&1 | tee /tmp/crew-ci-ws.log | tail -5; then
        WS_PASS=$(grep "^test result:" /tmp/crew-ci-ws.log | awk '{for(i=1;i<=NF;i++){if($i~/passed/){gsub(/[^0-9]/,"",$i);p+=$i}}}END{print p+0}')
        WS_FAIL=$(grep "^test result:" /tmp/crew-ci-ws.log | awk '{for(i=1;i<=NF;i++){if($i~/failed/){gsub(/[^0-9]/,"",$i);f+=$i}}}END{print f+0}')
        if [ "${WS_FAIL:-0}" -eq 0 ]; then
            pass "workspace tests ($WS_PASS passed)"
        else
            fail "workspace tests ($WS_FAIL failures)"
        fi
    else
        fail "workspace tests"
    fi

    # Focused subsystem tests (beyond workspace, with feature flags)
    if [ "$QUICK" = false ]; then
        section "Subsystem Tests"

        echo "  crew-cli with API feature..."
        if cargo test -p crew-cli --features api 2>&1 | tail -3; then
            pass "crew-cli --features api"
        else
            fail "crew-cli --features api"
        fi
    fi
fi

# ── 4. Build check (quick mode skips this) ────────────────────────────
if [ "$QUICK" = false ] && [ -z "$SUBSYSTEM" ]; then
    section "Build Check"
    if cargo build --workspace 2>&1 | tail -3; then
        pass "workspace build"
    else
        fail "workspace build"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────
section "Summary"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "CI FAILED: $FAIL check(s) failed."
    exit 1
else
    echo "All checks passed."
    exit 0
fi
