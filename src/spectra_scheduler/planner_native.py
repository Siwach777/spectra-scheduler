"""Optional compiled planning kernel; explicit loading, shared checked NumPy buffers."""

import argparse
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=4)
def _load(path):
    library = ctypes.CDLL(str(Path(path).resolve()))
    library.spectra_planner_version.restype = ctypes.c_int64
    if library.spectra_planner_version() != 1:
        raise ValueError("unsupported native planning interface")
    function = library.spectra_plan
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    function.argtypes = [ctypes.c_int64] * 4 + [
        doubles, integers, integers, integers, doubles, doubles
    ]
    function.restype = None
    return function


def kernel():
    path = os.environ.get("SPECTRA_PLANNER_LIBRARY")
    return _load(path) if path else None


def runtime_details():
    path = os.environ.get("SPECTRA_PLANNER_LIBRARY")
    if not path:
        return {"planner": "numpy", "planner_library_sha256": None}
    kernel()
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"planner": "native", "planner_library_sha256": digest}


def build(output):
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("a C++17 compiler is required for the optional planning kernel")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=output.parent, suffix=".so")
    os.close(descriptor)
    temporary = Path(name)
    try:
        subprocess.run([
            compiler, "-std=c++17", "-O3", "-shared", "-fPIC",
            str(Path(__file__).with_name("_planner.cpp")), "-o", str(temporary),
        ], check=True)
        _load(str(temporary))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/timing_planner.so"))
    args = parser.parse_args(arguments)
    print(build(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
