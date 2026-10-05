#!/usr/bin/env python3
"""Patch a mamba_ssm / causal_conv1d sdist setup.py for a Hopper-only build.

  1. sm90-only codegen  -- neither setup.py exposes a CUDA-arch env var, so they
     would otherwise emit sm75..sm121 (9 archs) for every .cu file.
  2. -std=c++17 -> c++20 -- torch>=2.14 ATen headers hard-error on C++17
     ("#error C++20 or later compatible compiler is required to use ATen").

Idempotent: re-running on an already-patched tree is a no-op.
"""
import sys

ANCHOR = "    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as"
SM90 = (
    '    # dm: Hopper-only build -- restrict to sm_90 to cut compile time\n'
    '    if not HIP_BUILD:\n'
    '        cc_flag = ["-gencode", "arch=compute_90,code=sm_90"]\n\n'
)

def main(path: str) -> None:
    src = open(path).read()
    out = src
    if "dm: Hopper-only" not in out:
        if out.count(ANCHOR) != 1:
            raise SystemExit(f"{path}: expected exactly 1 anchor, found {out.count(ANCHOR)}")
        out = out.replace(ANCHOR, SM90 + ANCHOR, 1)
    out = out.replace("-std=c++17", "-std=c++20")
    if out != src:
        open(path, "w").write(out)
        print(f"patched {path}")
    else:
        print(f"already patched {path}")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__ + "\n  usage: _patch_kernel_sdist.py <sdist>/setup.py"); sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    main(sys.argv[1])
