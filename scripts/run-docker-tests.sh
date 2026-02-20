#!/usr/bin/env bash
set -euo pipefail

# Docker test runner — single container, all suites
# Usage: ./scripts/run-docker-tests.sh [suite...]
#   No args = run all 6 Docker suites
#   Args   = run named suites (cowork, runtime, tutorial, edge, invariant, workspace)

IMAGE="${DOLFINX_MCP_IMAGE:-dolfinx-mcp:latest}"
CONTAINER_NAME="dolfinx-mcp-tests-$$"

# Map suite name → pytest path (portable, no bash 4 associative arrays)
suite_path() {
  case "$1" in
    cowork)    echo "/app/tests/test_cowork_fixes.py" ;;
    runtime)   echo "/app/tests/test_runtime_contracts.py" ;;
    tutorial)  echo "/app/tests/test_tutorial_workflows.py" ;;
    edge)      echo "/app/tests/test_edge_case_contracts.py" ;;
    invariant) echo "/app/tests/test_contracts.py -k invariant" ;;
    workspace) echo "/app/tests/test_workspace_tools.py" ;;
    *) return 1 ;;
  esac
}

ALL_SUITES="cowork runtime tutorial edge invariant workspace"

# Default: all suites
if [ $# -eq 0 ]; then
  selected=($ALL_SUITES)
else
  selected=("$@")
fi

cleanup() {
  echo "Cleaning up container ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
}
trap cleanup EXIT

# Start persistent container (sleep process keeps it alive)
echo "Starting test container (${IMAGE})..."
docker create --name "${CONTAINER_NAME}" \
  --entrypoint sleep "${IMAGE}" 3600 >/dev/null
docker start "${CONTAINER_NAME}" >/dev/null

PASSED=0
FAILED=0
RESULTS=()

for suite in "${selected[@]}"; do
  pytest_args=$(suite_path "$suite" 2>/dev/null) || {
    echo "Unknown suite: ${suite} (valid: ${ALL_SUITES})"
    exit 1
  }

  echo ""
  echo "=== Running: ${suite} ==="

  # shellcheck disable=SC2086
  if docker exec "${CONTAINER_NAME}" python -m pytest ${pytest_args} -v; then
    RESULTS+=("pass ${suite}")
    ((PASSED++))
  else
    RESULTS+=("FAIL ${suite}")
    ((FAILED++))
  fi
done

# Summary
echo ""
echo "================================"
echo "Docker Test Summary"
echo "================================"
for r in "${RESULTS[@]}"; do
  echo "  ${r}"
done
echo "--------------------------------"
echo "  Passed: ${PASSED}  Failed: ${FAILED}"
echo "================================"

[ "${FAILED}" -eq 0 ] || exit 1
