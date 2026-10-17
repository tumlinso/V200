#!/usr/bin/env bash
set -euo pipefail
nsys_target=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/profilers/12.9/Nsight_Systems_2025.3/target-linux-x64/nsys
if [[ "${1:-}" == profile ]]; then
  shift
  exec "$nsys_target" profile --cuda-graph-trace=node --cpuctxsw=none --wait=all "$@"
fi
exec "$nsys_target" "$@"
