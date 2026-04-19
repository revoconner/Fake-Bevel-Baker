"""Build the MikkTSpace ctypes shared library.

Compiles `mikkt_wrapper.c` together with the reference `mikktspace.c`
(copied from files_from_blender/) into `mikkt.dll` (Windows) or
`libmikkt.so` (POSIX). Run once per environment; re-run if any of the C
sources change.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLENDER_FILES = HERE.parents[1] / "files_from_blender"


def _out_name() -> str:
    if sys.platform == "win32":
        return "mikkt.dll"
    if sys.platform == "darwin":
        return "libmikkt.dylib"
    return "libmikkt.so"


def _find_compiler() -> list[str]:
    # gcc first on Windows: clang defaults to MSVC target and needs the MSVC SDK.
    order = ("gcc", "clang") if sys.platform == "win32" else ("clang", "gcc")
    for name in order:
        path = shutil.which(name)
        if path:
            return [path]
    raise RuntimeError("No C compiler found (looked for gcc, clang).")


def build() -> Path:
    if not BLENDER_FILES.exists():
        raise FileNotFoundError(BLENDER_FILES)
    mikkt_c = BLENDER_FILES / "mikktspace.c"
    mikkt_h = BLENDER_FILES / "mikktspace.h"
    wrapper_c = HERE / "mikkt_wrapper.c"
    if not mikkt_c.exists() or not mikkt_h.exists():
        raise FileNotFoundError(f"Missing {mikkt_c} or {mikkt_h}")

    out = HERE / _out_name()
    cmd = _find_compiler() + [
        "-O2",
        "-shared",
        f"-I{BLENDER_FILES}",
        str(wrapper_c),
        str(mikkt_c),
        "-o",
        str(out),
        "-lm",
    ]
    # -fPIC is only meaningful on POSIX; Windows DLLs don't need it.
    if sys.platform != "win32":
        cmd.insert(-3, "-fPIC")
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("stdout:", r.stdout)
        print("stderr:", r.stderr)
        raise RuntimeError(f"compiler failed with code {r.returncode}")
    print("Built:", out)
    return out


if __name__ == "__main__":
    build()
