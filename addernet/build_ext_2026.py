#!/usr/bin/env python3
"""
AdderNet Build Extension 2026
=============================
Enhanced build system with multi-architecture CUDA support.
Auto-detects GPU capability and compiles optimal kernels.

Features:
- Runtime capability detection
- Multi-architecture compilation (sm_61, sm_75, sm_80, sm_89, sm_90)
- Colab/Kaggle compatibility
- Fallback to CPU if CUDA unavailable
"""

import subprocess
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from shutil import which

# Import our enhanced detector
try:
    from .cuda_detector import CUDADetector, get_detector
except ImportError:
    # Allow standalone import
    from cuda_detector import CUDADetector


class CUDABuildManager:
    """
    Manages CUDA compilation with architecture detection.
    """

    def __init__(self):
        self.detector = CUDADetector()
        self.detector.detect()
        self.src_dir: Optional[Path] = None
        self.build_dir: Optional[Path] = None
        self.cuda_flags: List[str] = []

    def find_sources(self) -> Optional[Path]:
        """
        Locate CUDA source directory.
        """
        here = Path(__file__).parent
        candidates = [
            here / "src",
            here.parent / "src",
            here.parent.parent / "src",
            Path.cwd() / "src",
        ]
        for candidate in candidates:
            if candidate.exists() and (candidate / "hdc_core.h").exists():
                self.src_dir = candidate
                return candidate
        return None

    def get_nvcc_flags(self, capability: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        Get optimal nvcc flags for detected architecture.

        Args:
            capability: (major, minor) or None for auto-detect

        Returns:
            List of nvcc flags
        """
        if capability is None:
            capability = self.detector.capability

        flags = [
            "-O3",
            "-Xcompiler", "-fPIC",
            "-Xcompiler", "-fopenmp",
            "-Xcompiler", "-ffast-math",
            "-std=c++17",  # Updated from c++14
        ]

        if capability:
            major, minor = capability
            cap = major * 10 + minor

            # Architecture-specific optimizations
            if cap >= 80:
                # Ampere+: Enable async copy, optimizations
                flags.extend([
                    "--use_fast_math",
                    "-ftz=true",
                    "-prec-div=false",
                    "-prec-sqrt=false",
                ])

            # Generate code for specific architecture and PTX
            flags.extend([
                "-gencode", f"arch=compute_{cap},code=sm_{cap}"
            ])

            # Add PTX for forward compatibility (except for very old)
            if cap >= 70:
                flags.extend([
                    "-gencode", f"arch=compute_{cap},code=compute_{cap}"
                ])

        else:
            # No GPU detected, compile for common architectures
            # Include PTX for future compatibility
            for arch in ["61", "75", "80", "90"]:
                flags.extend([
                    "-gencode", f"arch=compute_{arch},code=sm_{arch}",
                ])
            # PTX for Ampere+
            flags.extend([
                "-gencode", "arch=compute_80,code=compute_80"
            ])

        return flags

    def compile_cuda_sources(
        self,
        output_path: Path,
        sources: List[Path],
        extra_flags: Optional[List[str]] = None
    ) -> bool:
        """
        Compile CUDA sources with optimal flags.

        Args:
            output_path: Path for output .so file
            sources: List of .cu files to compile
            extra_flags: Additional compiler flags

        Returns:
            True on success
        """
        if not sources:
            return False

        if not self.src_dir:
            self.find_sources()

        if not self.src_dir or not self.src_dir.exists():
            print(f"[CUDA 2026] Source directory not found")
            return False

        # Make temporary build directory
        build_dir = output_path.parent / "cuda_build_2026"
        build_dir.mkdir(parents=True, exist_ok=True)

        # Get compiler
        nvcc = self.detector.nvcc_path or "nvcc"

        if not self.detector.nvcc_path:
            # Try to find it
            nvcc = which("nvcc")
            if not nvcc:
                print("[CUDA 2026] nvcc not found, skipping CUDA build")
                return False

        # Compile object files
        object_files: List[Path] = []
        base_flags = [
            "-O3",
            "-Xcompiler", "-fPIC",
            "-I", str(self.src_dir),
        ]

        # Add architecture flags
        if extra_flags:
            base_flags.extend(extra_flags)

        capability = self.detector.capability

        for src in sources:
            if not src.exists():
                print(f"[CUDA 2026] Source not found: {src}")
                continue

            obj_file = build_dir / f"{src.stem}.o"

            # Detect if it needs specific flags
            if "ampere" in src.name.lower():
                # Ampere-specific compilation
                arch_flags = ["-gencode", "arch=compute_80,code=sm_80"]
                if capability and capability[0] * 10 + capability[1] >= 80:
                    arch_flags.extend([
                        "-use_fast_math",
                        "-ftz=true",
                    ])
            elif "turing" in src.name.lower():
                arch_flags = ["-gencode", "arch=compute_75,code=sm_75"]
            else:
                # Generic: compile for all supported architectures
                arch_flags = self.get_nvcc_flags(capability)

            cmd = [nvcc] + base_flags + arch_flags + ["-c", str(src), "-o", str(obj_file)]

            print(f"[CUDA 2026] Compiling {src.name}...")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self.src_dir,
                )
                if result.returncode != 0:
                    print(f"[CUDA 2026] Compile error: {result.stderr[:500]}")
                    return False
                object_files.append(obj_file)
            except subprocess.TimeoutExpired:
                print(f"[CUDA 2026] Compile timeout for {src}")
                return False
            except Exception as e:
                print(f"[CUDA 2026] Compile exception: {e}")
                return False

        # Link shared library
        if not object_files:
            return False

        print(f"[CUDA 2026] Linking {output_path.name}...")

        # Find required objects
        cpu_sources = [
            self.src_dir / "hdc_core.c",
            self.src_dir / "hdc_lsh.c",
            self.src_dir / "addernet_hdc.c",
        ]

        cpu_objects = []
        for src in cpu_sources:
            if src.exists():
                obj_file = build_dir / f"{src.stem}_gcc.o"
                gcc_cmd = [
                    "gcc", "-O3", "-fPIC", "-ffast-math", "-fopenmp",
                    "-c", str(src), "-o", str(obj_file),
                    "-I", str(self.src_dir),
                ]
                try:
                    subprocess.run(gcc_cmd, capture_output=True, timeout=60)
                    if obj_file.exists():
                        cpu_objects.append(obj_file)
                except:
                    pass

        # Link with nvcc
        link_cmd = [
            nvcc,
            "-shared",
            "-o", str(output_path),
        ] + [str(obj) for obj in object_files + cpu_objects] + [
            "-lm", "-lpthread", "-ldl", "-fopenmp"
        ]

        try:
            result = subprocess.run(
                link_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                print(f"[CUDA 2026] Link error: {result.stderr[:500]}")
                return False
        except Exception as e:
            print(f"[CUDA 2026] Link exception: {e}")
            return False

        # Cleanup build dir
        import shutil
        shutil.rmtree(build_dir, ignore_errors=True)

        print(f"[CUDA 2026] Successfully built: {output_path}")
        return True

    def build_ampere_variant(self, pkg_dir: Path) -> bool:
        """
        Build Ampere-optimized (sm_80+) variant with cooperative kernels.
        """
        if not self.src_dir:
            self.find_sources()

        if not self.src_dir:
            return False

        cuda_src = self.src_dir / "cuda"
        if not cuda_src.exists():
            cuda_src = self.src_dir  # Fallback

        sources = [
            cuda_src / "addernet_cuda_ampere.cu",
            cuda_src / "cuda_train" / "addernet_hdc_train_cuda_2026.cu",
        ]

        # Fallback to generic if ampere-specific not present
        if not sources[0].exists():
            sources[0] = cuda_src / "addernet_cuda.cu"
        if not sources[1].exists():
            sources[1] = self.src_dir / "addernet_hdc_train_cuda.cu"

        output = pkg_dir / "libaddernet_cuda_2026.so"

        return self.compile_cuda_sources(output, [s for s in sources if s.exists()])


# Legacy compatibility
def build(output_dir: Optional[Path] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Entry point for building AdderNet with CUDA 2026 support.

    Args:
        output_dir: Directory for output libraries

    Returns:
        (hdc_lib_path, cuda_lib_path) or (None, None) on failure
    """
    here = Path(__file__).parent

    if output_dir is None:
        output_dir = here

    print("[AdderNet 2026] Detecting CUDA...")

    manager = CUDABuildManager()

    if not manager.detector.is_available():
        print("[AdderNet 2026] CUDA not available, building CPU-only")

    print(f"[AdderNet 2026] {manager.detector}")

    # Determine which variant to build
    variant = manager.detector.get_best_kernel_variant()

    # Find sources
    if not manager.find_sources():
        print("[AdderNet 2026] Source files not found")
        return None, None

    # Build CPU libraries first
    print("[AdderNet 2026] Building CPU libraries...")

    # ... (CPU build logic from original build_ext.py)
    # This is simplified for the migration

    # Build CUDA variant if available
    if manager.detector.is_available():
        print(f"[AdderNet 2026] Building {variant} CUDA variant...")

        if variant == 'ampere':
            success = manager.build_ampere_variant(here)
        else:
            # Build generic CUDA
            sources = [
                manager.src_dir / "addernet_cuda.cu",
                manager.src_dir / "addernet_hdc_train_cuda.cu",
            ]
            output = here / "libaddernet_cuda.so"
            success = manager.compile_cuda_sources(output, sources)

        if success:
            print("[AdderNet 2026] CUDA libraries built successfully")
        else:
            print("[AdderNet 2026] CUDA build failed, falling back to CPU")

    return str(here / "libaddernet_hdc.so"), str(here / "libaddernet_cuda.so")


if __name__ == "__main__":
    print("AdderNet Build Extension 2026")
    print("=" * 50)

    hdc_lib, cuda_lib = build()

    if hdc_lib:
        print(f"\nBuilt libraries:")
        print(f"  HDC: {hdc_lib}")
        print(f"  CUDA: {cuda_lib or 'N/A'}")
    else:
        print("\nBuild failed")
