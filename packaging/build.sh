#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BUILD_DIR="$ROOT/build-multipair"
CUDA_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9"

if [[ "$BUILD_DIR" == "$ROOT/build" || -L "$BUILD_DIR" ]]; then
  echo "refusing unsafe build directory: $BUILD_DIR" >&2
  exit 2
fi
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "HPC CUDA 12.9 nvcc not found at $CUDA_ROOT/bin/nvcc" >&2
  exit 2
fi
if [[ ! -x /usr/bin/g++ ]]; then
  echo "C++ compiler not found at /usr/bin/g++" >&2
  exit 2
fi
if ! "$CUDA_ROOT/bin/nvcc" --version | grep -q 'release 12\.9'; then
  echo "toolkit at $CUDA_ROOT is not CUDA 12.9" >&2
  exit 2
fi
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cache_root="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt" | tail -n 1)"
  if [[ "$cache_root" != "$ROOT" ]]; then
    echo "refusing build cache from another checkout: $cache_root" >&2
    exit 2
  fi
fi

echo "Building V200 multipair CPU/runtime artifacts in $BUILD_DIR with HPC CUDA 12.9."
echo "This script does not initialize a GPU, download a model, start a service, or install system files."
cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DV200_CUDA_TOOLKIT_ROOT="$CUDA_ROOT" \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE= \
  -DBUILD_TESTING=ON \
  -DV200_BUILD_BOOTSTRAP_EXAMPLES=OFF \
  -DV200_BUILD_CUDA_PROBE=OFF \
  -DV200_ENABLE_GPU_TESTS=OFF \
  -DV200_INTEGRATION_BUILD_GPU_SMOKE=OFF \
  -DV200_EXECUTION_BUILD_GPU_SMOKE=OFF
cmake --build "$BUILD_DIR" --parallel 2
ctest --test-dir "$BUILD_DIR" --output-on-failure
python3 "$ROOT/packaging/test_deployment.py"
