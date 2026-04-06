#!/usr/bin/env python3
"""
AdderNet CUDA 2026 Detection System
====================================
Enhanced detection for Colab, Kaggle, and other cloud environments.
Implements A+C approach: Multiple paths + pip/conda package detection.
"""

import os
import sys
import glob
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict


class CUDADetector:
    """
    Detects CUDA installation across multiple environments:
    - System installations
    - Conda environments
    - pip-installed CUDA packages
    - Cloud platforms (Colab, Kaggle)
    """

    # Priority order for nvcc discovery
    NVCC_PATHS = [
        # System-wide installations
        "/usr/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-*/bin/nvcc",  # Versioned installs
        "/opt/cuda/bin/nvcc",
        "/opt/cuda-*/bin/nvcc",
        "/usr/local/nvidia/cuda/bin/nvcc",

        # Environment variable
        "$CUDA_HOME/bin/nvcc",
        "$CUDA_PATH/bin/nvcc",

        # Conda environments
        "$CONDA_PREFIX/bin/nvcc",
        "$CONDA_PREFIX/pkgs/cuda-*/bin/nvcc",
        "$HOME/miniconda3/bin/nvcc",
        "$HOME/anaconda3/bin/nvcc",
        "/opt/conda/bin/nvcc",

        # Colab-specific
        "/usr/local/cuda-12.*/bin/nvcc",
        "/usr/local/cuda-11.*/bin/nvcc",

        # Kaggle-specific
        "/opt/conda/envs/*/bin/nvcc",
    ]

    # Library names to search for runtime
    # IMPORTANT: libcuda.so.1 (driver) must come before libcudart.so (runtime)
    # because we need the driver for cuInit/cuDeviceGet via ctypes
    CUDA_LIBS = [
        "libcuda.so.1",
        "libcuda.so",
        "libcudart.so.12",
        "libcudart.so.11",
        "libcudart.so",
    ]

    def __init__(self):
        self.nvcc_path: Optional[str] = None
        self.cuda_home: Optional[str] = None
        self.cuda_version: Optional[str] = None
        self.libcuda_path: Optional[str] = None
        self.capability: Optional[Tuple[int, int]] = None
        self.gpu_name: Optional[str] = None
        self._detected = False

    def detect(self) -> bool:
        """
        Run full CUDA detection sequence.

        Returns:
            True if CUDA is available and usable
        """
        if self._detected:
            return self.nvcc_path is not None

        self._detected = True

        # Try multiple detection strategies
        if self._detect_nvcc():
            self._detect_cuda_home()
            self._detect_version()

        # Also detect runtime library (even if nvcc missing)
        self._detect_runtime_lib()

        # Try to detect GPU capability via ctypes
        self._detect_gpu_capability()

        return self.nvcc_path is not None or self.libcuda_path is not None

    def _expand_path(self, path: str) -> List[str]:
        """Expand environment variables and globs in paths."""
        expanded = os.path.expandvars(path)
        if '*' in expanded:
            return sorted(glob.glob(expanded))
        return [expanded] if expanded else []

    def _detect_nvcc(self) -> bool:
        """
        Find nvcc in multiple locations.

        Returns:
            True if nvcc found
        """
        # First check PATH
        from shutil import which
        nvcc_in_path = which("nvcc")
        if nvcc_in_path and self._validate_nvcc(nvcc_in_path):
            self.nvcc_path = nvcc_in_path
            return True

        # Check common paths
        for pattern in self.NVCC_PATHS:
            for path in self._expand_path(pattern):
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    if self._validate_nvcc(path):
                        self.nvcc_path = path
                        return True

        # Try pip/conda packages
        if self._detect_from_pip_packages():
            return True

        return False

    def _validate_nvcc(self, path: str) -> bool:
        """Verify nvcc is functional."""
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 and "NVIDIA" in result.stdout
        except:
            return False

    def _detect_from_pip_packages(self) -> bool:
        """
        Detect CUDA from pip-installed packages (nvidia-cuda-runtime-cu12, etc).
        Common in Colab/Kaggle after 'pip install nvidia-cuda-runtime-cu12'
        """
        try:
            import importlib.util

            # Check for nvidia packages
            for pkg_name in ["nvidia.cuda_runtime", "nvidia.cuda_nvcc"]:
                spec = importlib.util.find_spec(pkg_name)
                if spec and spec.origin:
                    pkg_dir = Path(spec.origin).parent
                    # Look for nvcc in package
                    nvcc_candidates = [
                        pkg_dir / "bin" / "nvcc",
                        pkg_dir / "../bin/nvcc",
                        pkg_dir / "../../../../../bin/nvcc",
                    ]
                    for nvcc in nvcc_candidates:
                        if nvcc.exists():
                            self.nvcc_path = str(nvcc)
                            self.cuda_home = str(nvcc.parent.parent)
                            return True

            # Check site-packages directly
            import site
            for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
                if not site_dir:
                    continue
                site_path = Path(site_dir)

                # Look for nvidia-cuda-nvcc
                nvcc_patterns = [
                    site_path / "nvidia"/ "cuda_nvcc" / "bin" / "nvcc",
                    site_path / "nvidia_cuda_nvcc" / "bin" / "nvcc",
                ]

                # Also try nvidia packages with version
                for pattern in site_path.glob("nvidia_cuda_nvcc_cu*/bin/nvcc"):
                    nvcc_patterns.append(pattern)

                for nvcc_path in nvcc_patterns:
                    if nvcc_path.exists():
                        self.nvcc_path = str(nvcc_path)
                        self.cuda_home = str(nvcc_path.parent.parent)
                        return True

        except Exception as e:
            pass

        return False

    def _detect_cuda_home(self):
        """Determine CUDA_HOME from nvcc location."""
        if self.nvcc_path:
            # nvcc is at $CUDA_HOME/bin/nvcc
            self.cuda_home = str(Path(self.nvcc_path).parent.parent)
        elif "CUDA_HOME" in os.environ:
            self.cuda_home = os.environ["CUDA_HOME"]
        elif "CUDA_PATH" in os.environ:
            self.cuda_home = os.environ["CUDA_PATH"]

    def _detect_version(self):
        """Extract CUDA version from nvcc."""
        if not self.nvcc_path:
            return

        try:
            result = subprocess.run(
                [self.nvcc_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Parse "release X.Y" from output
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'release' and i + 1 < len(parts):
                            self.cuda_version = parts[i + 1].rstrip(',')
                            return
        except:
            pass

    def _detect_runtime_lib(self):
        """Find libcuda.so.1 (driver) — required for GPU capability detection."""
        # CRITICAL: Check system paths for libcuda.so.1 FIRST.
        # libcuda.so.1 is the NVIDIA DRIVER library, NOT the CUDA toolkit.
        # It lives in /usr/lib/..., NOT in /usr/local/cuda/lib64/.
        # /usr/local/cuda/lib64/ only has libcudart.so (runtime), which lacks
        # cuInit/cuDeviceGet — we need the DRIVER for ctypes GPU detection.
        system_paths = [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/local/nvidia/lib64",   # Kaggle
            "/usr/local/nvidia/lib",
        ]

        for path in system_paths:
            full_path = os.path.join(path, "libcuda.so.1")
            if os.path.exists(full_path):
                self.libcuda_path = full_path
                return

        # Fallback: check LD_LIBRARY_PATH (may have libcudart but not libcuda)
        if "LD_LIBRARY_PATH" in os.environ:
            for path in os.environ["LD_LIBRARY_PATH"].split(':'):
                full_path = os.path.join(path, "libcuda.so.1")
                if os.path.exists(full_path):
                    self.libcuda_path = full_path
                    return

        # Last resort: accept libcudart.so (won't support ctypes GPU detection)
        for lib_name in ["libcudart.so.12", "libcudart.so.11", "libcudart.so"]:
            if "LD_LIBRARY_PATH" in os.environ:
                for path in os.environ["LD_LIBRARY_PATH"].split(':'):
                    full_path = os.path.join(path, lib_name)
                    if os.path.exists(full_path):
                        self.libcuda_path = full_path
                        return

            for path_template in ["/usr/local/cuda/lib64", "/usr/local/cuda/lib"]:
                for path in self._expand_path(path_template):
                    full_path = os.path.join(path, lib_name)
                    if os.path.exists(full_path):
                        self.libcuda_path = full_path
                        return

    def _detect_gpu_capability(self):
        """
        Detect GPU compute capability via ctypes and CUDA driver API.
        This works even without nvcc if libcuda.so is available.
        """
        if not self.libcuda_path:
            # Try standard locations
            for lib_path in ["/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                           "/usr/lib64/libcuda.so.1"]:
                if os.path.exists(lib_path):
                    self.libcuda_path = lib_path
                    break

        if not self.libcuda_path:
            return

        try:
            import ctypes

            # Load CUDA driver library
            cuda = ctypes.CDLL(self.libcuda_path)

            # CUDA driver types
            CUresult = ctypes.c_int
            CUdevice = ctypes.c_int

            # Required functions
            cuInit = cuda.cuInit
            cuInit.argtypes = [ctypes.c_uint]
            cuInit.restype = CUresult

            cuDeviceGet = cuda.cuDeviceGet
            cuDeviceGet.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]
            cuDeviceGet.restype = CUresult

            cuDeviceGetAttribute = cuda.cuDeviceGetAttribute
            cuDeviceGetAttribute.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,  # attrib
                CUdevice
            ]
            cuDeviceGetAttribute.restype = CUresult

            cuDeviceGetName = cuda.cuDeviceGetName
            cuDeviceGetName.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                CUdevice
            ]
            cuDeviceGetName.restype = CUresult

            # Initialize CUDA
            result = cuInit(0)
            if result != 0:
                return

            # Get first device
            device = CUdevice()
            result = cuDeviceGet(ctypes.byref(device), 0)
            if result != 0:
                return

            # Get device name
            name_buf = ctypes.create_string_buffer(256)
            cuDeviceGetName(name_buf, 256, device)
            self.gpu_name = name_buf.value.decode('utf-8', errors='replace')

            # CUDA_DEVICE_ATTRIBUTE constants
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

            major = ctypes.c_int()
            minor = ctypes.c_int()

            cuDeviceGetAttribute(
                ctypes.byref(major),
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device
            )
            cuDeviceGetAttribute(
                ctypes.byref(minor),
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device
            )

            self.capability = (major.value, minor.value)

        except Exception as e:
            pass

    def get_capability_int(self) -> Optional[int]:
        """Get compute capability as single integer (e.g., 80 for sm_80)."""
        if self.capability:
            return self.capability[0] * 10 + self.capability[1]
        return None

    def get_arch_flags(self) -> List[str]:
        """
        Get nvcc architecture flags optimized for detected GPU.
        Includes PTX for forward compatibility.
        """
        if not self.capability:
            # Default: support common architectures
            return [
                "-gencode", "arch=compute_61,code=sm_61",
                "-gencode", "arch=compute_75,code=sm_75",
                "-gencode", "arch=compute_80,code=sm_80",
            ]

        major, minor = self.capability
        cap = major * 10 + minor

        flags = []

        # Add specific architecture
        flags.extend([
            "-gencode", f"arch=compute_{cap},code=sm_{cap}"
        ])

        # Add PTX for JIT compilation (forward compatibility)
        if cap >= 80:
            # Ampere+: include PTX for future architectures
            flags.extend([
                "-gencode", f"arch=compute_{cap},code=compute_{cap}"
            ])

        return flags

    def get_best_kernel_variant(self) -> str:
        """
        Determine optimal kernel variant for detected GPU.

        Returns:
            'ampere' | 'turing' | 'legacy'
        """
        cap = self.get_capability_int()
        if cap is None:
            return 'legacy'

        if cap >= 80:
            return 'ampere'  # sm_80+: Tensor Cores, 100KB shared
        elif cap >= 70:
            return 'turing'  # sm_70-75: Unified memory, 64KB shared
        else:
            return 'legacy'  # sm_61 and below

    def to_dict(self) -> Dict:
        """Export detection results as dictionary."""
        return {
            'nvcc_path': self.nvcc_path,
            'cuda_home': self.cuda_home,
            'cuda_version': self.cuda_version,
            'libcuda_path': self.libcuda_path,
            'capability': self.capability,
            'capability_int': self.get_capability_int(),
            'gpu_name': self.gpu_name,
            'kernel_variant': self.get_best_kernel_variant(),
            'arch_flags': self.get_arch_flags(),
            'available': self.is_available(),
        }

    def is_available(self) -> bool:
        """Check if CUDA is available for use."""
        return self.nvcc_path is not None or self.libcuda_path is not None

    def __str__(self) -> str:
        """Pretty print detection results."""
        if not self.is_available():
            return "CUDA: Not detected"

        lines = ["CUDA Detection Results:", "-" * 40]
        lines.append(f"  NVCC: {self.nvcc_path or 'Not found'}")
        lines.append(f"  Version: {self.cuda_version or 'Unknown'}")
        lines.append(f"  CUDA_HOME: {self.cuda_home or 'Not set'}")
        lines.append(f"  GPU: {self.gpu_name or 'Unknown'}")
        if self.capability:
            lines.append(f"  Capability: sm_{self.capability[0]}{self.capability[1]}")
        lines.append(f"  Kernel: {self.get_best_kernel_variant()}")
        lines.append(f"  Library: {self.libcuda_path or 'Not found'}")
        return '\n'.join(lines)


# Convenience singleton
def get_detector() -> CUDADetector:
    """Get singleton detector instance."""
    if not hasattr(get_detector, '_instance'):
        get_detector._instance = CUDADetector()
        get_detector._instance.detect()
    return get_detector._instance


# Test when run directly
if __name__ == "__main__":
    print("AdderNet CUDA 2026 Detection System")
    print("=" * 50)

    detector = CUDADetector()
    detector.detect()

    print(detector)
    print()

    if detector.is_available():
        print("\nDetailed information:")
        import json
        print(json.dumps(detector.to_dict(), indent=2))
    else:
        print("\nCUDA not detected. Checked paths:")
        for pattern in CUDADetector.NVCC_PATHS[:10]:
            for path in detector._expand_path(pattern):
                print(f"  - {path}: {'EXISTS' if os.path.exists(path) else 'NOT FOUND'}")
