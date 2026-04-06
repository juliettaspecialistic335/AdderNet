import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent

# CUDA 2026 Detection System
try:
    from .cuda_detector import CUDADetector
    _cuda_detector = CUDADetector()
    _cuda_detector.detect()
    if _cuda_detector.is_available:
        print(f"[AdderNet 2026] {_cuda_detector}")
except Exception:
    _cuda_detector = None

if sys.platform == "darwin":
    _lib_hdc_name = "libaddernet_hdc.dylib"
    _lib_addernet_name = "libaddernet.dylib"
    _lib_cuda_name = "libaddernet_cuda.dylib"
elif sys.platform == "win32":
    _lib_hdc_name = "addernet_hdc.dll"
    _lib_addernet_name = "addernet.dll"
    _lib_cuda_name = "addernet_cuda.dll"
else:
    _lib_hdc_name = "libaddernet_hdc.so"
    _lib_addernet_name = "libaddernet.so"
    _lib_cuda_name = "libaddernet_cuda.so"

_need_build = False
if not (_HERE / _lib_hdc_name).exists() or not (_HERE / _lib_addernet_name).exists():
    _need_build = True

# Check for CUDA 2026 variant
_lib_cuda_2026 = _HERE / "libaddernet_cuda_2026.so"
_has_cuda_2026 = _lib_cuda_2026.exists()

if _need_build:
    try:
        # Try new 2026 build system first
        from . import build_ext_2026
        build_ext_2026.build()
    except Exception as e:
        # Fallback to legacy build
        from . import build_ext
        build_ext.build()

from .addernet import AdderNetLayer
from .addernet_hdc import AdderNetHDC, hdc_detect_backend
from .cluster import AdderCluster
from .boost import AdderBoost
from .attention import AdderAttention

# Export detection info
if _cuda_detector:
    def get_cuda_info():
        """Get CUDA detection information."""
        return _cuda_detector.to_dict()

AnHdcModel = AdderNetHDC

__version__ = "1.4.0"
__all__ = ["AdderNetLayer", "AdderNetHDC", "AnHdcModel", "hdc_detect_backend",
           "AdderCluster", "AdderBoost", "AdderAttention"]
