#!/usr/bin/env python3
"""Run the unchanged CUDA controller with its legacy identity import bridged.

Use the canonical Project Control Python environment. This does not replace
controller admission or manufacture a lease. The controller path is an explicit
argument so its receipt owner remains identifiable in /proc command lines.
"""

from pathlib import Path
import runpy
import sys
import types


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_controller.py /absolute/path/cuda_controller.py <arguments>")
    controller = Path(sys.argv[1]).resolve(strict=True)
    from project_control.runtime_identity import bind_runtime, validate_runtime

    identity = bind_runtime()
    validate_runtime(identity)
    expected = identity.skills_root / "cuda/scripts/cuda_controller.py"
    if controller != expected.resolve(strict=True):
        raise SystemExit("controller must be the CUDA script in the configured Skills root")

    def bind_canonical_runtime():
        current = bind_runtime()
        validate_runtime(current)
        return current

    # The installed controller imports this one historical symbol. Delegate
    # entirely to today's fail-closed identity verifier; do not clone it.
    legacy = types.ModuleType("coding_workflow_mcp")
    legacy.__path__ = []
    bridge = types.ModuleType("coding_workflow_mcp.runtime_identity")
    bridge.bind_canonical_runtime = bind_canonical_runtime
    legacy.runtime_identity = bridge
    sys.modules[legacy.__name__] = legacy
    sys.modules[bridge.__name__] = bridge
    sys.path.insert(0, str(controller.parent))
    sys.argv = [str(controller), *sys.argv[2:]]
    runpy.run_path(str(controller), run_name="__main__")


if __name__ == "__main__":
    main()
