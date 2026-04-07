import os
import sys
from pathlib import Path
import shutil as _shutil

_HERE = Path(__file__).parent

# Verbose flag control (set to False to disable CUDA detection logs)
# Can be controlled via ADDERNET_VERBOSE environment variable (0/1)
_ADDERNET_VERBOSE = os.environ.get("ADDERNET_VERBOSE", "1") == "1"

def set_verbose(enabled: bool):
    """Enable or disable verbose logging output.
    
    Note: Must be called BEFORE importing AdderNetLayer/AdderNetHDC
    to prevent CUDA detection logs. Better to use ADDERNET_VERBOSE env var.
    """
    global _ADDERNET_VERBOSE
    _ADDERNET_VERBOSE = enabled
    # Also set env var for child modules to see
    os.environ["ADDERNET_VERBOSE"] = "1" if enabled else "0"

def is_verbose() -> bool:
    """Check if verbose logging is enabled."""
    return _ADDERNET_VERBOSE

# CUDA 2026 Detection System (lazy - only runs when explicitly requested)
_cuda_detector = None
_cuda_detection_ran = False

def _try_detect_cuda():
    """Lazy CUDA detection, only runs once."""
    global _cuda_detector, _cuda_detection_ran
    if _cuda_detection_ran:
        return
    _cuda_detection_ran = True
    try:
        from .cuda_detector import CUDADetector
        _cuda_detector = CUDADetector()
        _cuda_detector.detect()
        if is_verbose():
            print(f"[AdderNet 2026] {_cuda_detector}")
    except Exception:
        pass

if sys.platform == "darwin":
    _lib_hdc_name = "libaddernet_hdc.dylib"
    _lib_addernet_name = "libaddernet.dylib"
    _lib_cuda_name = "libaddernet_cuda.dylib"
    _lib_cuda_2026_name = "libaddernet_cuda_2026.dylib"
elif sys.platform == "win32":
    _lib_hdc_name = "addernet_hdc.dll"
    _lib_addernet_name = "addernet.dll"
    _lib_cuda_name = "addernet_cuda.dll"
    _lib_cuda_2026_name = "addernet_cuda_2026.dll"
else:
    _lib_hdc_name = "libaddernet_hdc.so"
    _lib_addernet_name = "libaddernet.so"
    _lib_cuda_name = "libaddernet_cuda.so"
    _lib_cuda_2026_name = "libaddernet_cuda_2026.so"


def _find_lib(lib_name, search_dirs):
    """Find a shared library in multiple directories."""
    for d in search_dirs:
        p = d / lib_name
        if p.exists():
            return p
    return None


def _try_copy_libs():
    """
    Try to find .so files in common locations and copy them into the
    package directory so ctypes can load them reliably.
    """
    libs_needed = [_lib_hdc_name, _lib_addernet_name]
    all_present = all((_HERE / lib).exists() for lib in libs_needed)
    if all_present:
        return True

    # Search locations in priority order
    search_dirs = [
        _HERE,                                        # inside package
        _HERE.parent / "build",                       # dev make build
        _HERE / "build",                              # package/build
        Path("/usr/local/lib"),                       # system install
        Path("/usr/lib"),
    ]

    copied_any = False
    for lib_name in libs_needed:
        found = _find_lib(lib_name, search_dirs)
        if found and found.parent != _HERE:
            try:
                _shutil.copy2(found, _HERE / lib_name)
                copied_any = True
            except Exception:
                pass

    # Also check for CUDA variants
    for cuda_name in [_lib_cuda_2026_name, _lib_cuda_name]:
        if (_HERE / cuda_name).exists():
            continue
        found = _find_lib(cuda_name, search_dirs)
        if found and found.parent != _HERE:
            try:
                _shutil.copy2(found, _HERE / cuda_name)
                copied_any = True
            except Exception:
                pass

    return copied_any


# Try to find existing libs before triggering a rebuild
_try_copy_libs()

_need_build = False
if not (_HERE / _lib_hdc_name).exists() or not (_HERE / _lib_addernet_name).exists():
    _need_build = True

# Check for CUDA 2026 variant
_lib_cuda_2026 = _HERE / _lib_cuda_2026_name
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
def get_cuda_info():
    """Get CUDA detection information."""
    _try_detect_cuda()
    if _cuda_detector:
        return _cuda_detector.to_dict()
    return None

AnHdcModel = AdderNetHDC

__version__ = "1.4.5"
__all__ = ["AdderNetLayer", "AdderNetHDC", "AnHdcModel", "hdc_detect_backend",
           "AdderCluster", "AdderBoost", "AdderAttention", "set_verbose", "is_verbose"]
