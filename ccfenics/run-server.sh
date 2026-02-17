#!/bin/bash
# DOLFINx MCP Server launcher for Desktop Extension format
# Requires: Docker installed and dolfinx-mcp image built
#
# Build the image first:
#   docker build -t dolfinx-mcp https://github.com/estanley/ccFenics-plugin.git

set -euo pipefail

WORKSPACE="${1:-${HOME}/dolfinx-workspace}"
mkdir -p "$WORKSPACE"

if ! command -v docker &>/dev/null; then
  echo "Error: Docker is not installed. Install Docker Desktop from https://docker.com" >&2
  exit 1
fi

if ! docker image inspect dolfinx-mcp &>/dev/null 2>&1; then
  echo "Error: dolfinx-mcp Docker image not found. Build it with:" >&2
  echo "  docker build -t dolfinx-mcp https://github.com/estanley/ccFenics-plugin.git" >&2
  exit 1
fi

exec docker run --rm -i \
  --network none \
  -v "${WORKSPACE}:/workspace" \
  dolfinx-mcp
