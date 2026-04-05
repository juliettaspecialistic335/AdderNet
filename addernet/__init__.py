import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent

if sys.platform == "darwin":
    _lib_hdc_name = "libaddernet_hdc.dylib"
    _lib_addernet_name = "libaddernet.dylib"
elif sys.platform == "win32":
    _lib_hdc_name = "addernet_hdc.dll"
    _lib_addernet_name = "addernet.dll"
else:
    _lib_hdc_name = "libaddernet_hdc.so"
    _lib_addernet_name = "libaddernet.so"

_need_build = False
if not (_HERE / _lib_hdc_name).exists() or not (_HERE / _lib_addernet_name).exists():
    _need_build = True

if _need_build:
    from . import build_ext
    build_ext.build()

from .addernet import AdderNetLayer
from .addernet_hdc import AdderNetHDC, hdc_detect_backend
from .cluster import AdderCluster
from .boost import AdderBoost
from .attention import AdderAttention

AnHdcModel = AdderNetHDC

__version__ = "1.0.9"
__all__ = ["AdderNetLayer", "AdderNetHDC", "AnHdcModel", "hdc_detect_backend",
           "AdderCluster", "AdderBoost", "AdderAttention"]
