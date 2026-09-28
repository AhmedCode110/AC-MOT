"""MAC-COMPAT drop-in for SparseTrack's compiled `pbcvt` module: calls the
verbatim GMC() C++ function (gmc_shim.cpp, OpenCV videostab) via ctypes."""
import ctypes
from pathlib import Path
import numpy as np

_lib = ctypes.CDLL(str(Path(__file__).with_name("libgmc_shim.dylib")))
_lib.gmc_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


def GMC(curr, prev, downscale):
    curr = np.ascontiguousarray(curr, dtype=np.uint8)
    h, w = curr.shape[:2]
    ch = curr.shape[2] if curr.ndim == 3 else 1
    out = np.zeros((3, 3), np.float32)
    pv = None
    if prev is not None:
        prev = np.ascontiguousarray(prev, dtype=np.uint8)
        pv = prev.ctypes.data
    _lib.gmc_c(curr.ctypes.data, pv, h, w, ch, int(downscale), out.ctypes.data)
    return out
