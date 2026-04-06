# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AdderNet** is a machine learning library that performs inference without floating-point multiplications. It uses table lookups (LUT) and integer addition instead of multiply-accumulate operations, targeting embedded systems without FPUs (ESP32, STM32, Raspberry Pi).

The library exposes four main Python classes:
- `AdderNetLayer` — Single-variable regression via LUT lookup
- `AdderNetHDC` — Multivariate classification using Hyperdimensional Computing (HDC)
- `AdderCluster` — Ensemble of `AdderNetLayer` with voting/mean/stacking strategies
- `AdderBoost` — Gradient boosting with `AdderNetLayer` base estimators

## Build System

The project uses a **Makefile** for C/CUDA compilation and **setuptools** for Python packaging.

### Essential Commands

```bash
# Compile C libraries (required before Python usage)
make all                    # Builds libaddernet.so and libaddernet_hdc.so
make addernet               # Build libaddernet.so only
make hdc                    # Build libaddernet_hdc.so only

# Optional CUDA backends
make cuda                   # Inline PTX variant (no nvcc needed, uses gcc)
make cuda_native            # Native nvcc variant (requires CUDA toolkit)

# Testing
make test                   # Builds and runs all C unit tests
make test_addernet          # Run AdderNetLayer C tests only
make test_hdc               # Run AdderNetHDC C tests only
python test_validation.py   # Python integration/validation tests
python test_attention.py    # Python attention mechanism tests
pytest -v                   # Run pytest suite (if pytest installed)

# Cleanup
make clean                  # Removes build/ directory

# Python installation (editable, for development)
pip install -e .
# Or build + install wheel
pip install .
# CLI tool (installed with package)
addernet-build              # Runtime C/CUDA compilation from CLI
```

### Build Architecture

- **setup.py** uses a custom `MakeBuildExt` command that invokes `make all` during pip install
- **setup.py** also defines a custom `bdist_wheel` command that forces `manylinux2014_x86_64` platform tag for Linux wheels
- Platform-specific SIMD is auto-detected: AVX2 (x86_64), NEON (ARM), or scalar fallback
- Shared libraries are copied from `build/` to `addernet/` package directory during install
- C/CUDA source files are bundled into the package at `addernet/src/` so runtime auto-build works on systems that install from wheel (e.g., Colab)
- If `.so` files are missing at runtime, `addernet/__init__.py` tries `build_ext_2026` first, then falls back to legacy `build_ext`
- The package includes a CLI entry point `addernet-build` (`addernet.build_ext:build`) for manual compilation
- Two CUDA variants: `make cuda` (inline PTX via gcc, no nvcc) and `make cuda_native` (requires nvcc, builds `libaddernet_cuda.so`)

## Code Architecture

### C Library Structure (`src/`)

```
src/
├── addernet.c/h         # Single-variable LUT layer (an_layer, an_train, an_predict)
├── addernet_hdc.c/h     # HDC multivariate classifier
├── hdc_core.c/h         # Hypervector operations (XOR, bundling, Hamming distance)
├── hdc_lsh.c/h          # Locality-Sensitive Hashing for HDC encoding
├── hdc_core_cuda.c      # CUDA kernels for HDC operations (inline PTX, no nvcc needed)
├── hdc_cuda_batch.c     # Batch prediction CUDA implementation
├── addernet_cuda.cu     # Native CUDA training kernels (requires nvcc)
└── addernet_hdc_train_cuda.cu  # CUDA AdaptHD retraining
```

### Python Bindings (`addernet/`)

```
addernet/
├── __init__.py          # Auto-builds C libs, CUDA 2026 detection, exports public API
├── addernet.py          # AdderNetLayer ctypes wrapper
├── addernet_hdc.py      # AdderNetHDC ctypes wrapper with GPU training support
├── cluster.py           # AdderCluster ensemble implementation
├── boost.py             # AdderBoost gradient boosting
├── attention.py         # AdderAttention mechanism
├── build_ext.py         # Runtime C/CUDA compilation helper (legacy)
├── build_ext_2026.py    # Runtime C/CUDA compilation helper (2026 system)
└── cuda_detector.py     # CUDA availability detection (GPU, nvcc, compiler)
```

## Key Design Principles

### AdderNetLayer (Single-Variable)

- **Inference**: `result = offset_table[(int_input + bias) & mask]` — one memory load, zero arithmetic
- **Training**: Trial-and-error directional search (no gradients/backprop). Adjusts table entries by `+/- lr` based on error direction
- **Data expansion**: Linear interpolation/extrapolation to fill the LUT for untrained input ranges
- **Table size**: Must be power of 2 (default 256) for efficient masking

### AdderNetHDC (Multivariate)

- **Encoding**: Each (variable, bin) pair generates a seed → deterministic hypervector via `hv_from_seed()`
- **OnlineHD**: Single-pass training with novelty-weighted bundling (prevents model saturation)
- **AdaptHD**: Iterative retraining using `an_hdc_retrain()` with per-bit count accumulation
- **Early-exit Hamming**: Stops distance calculation once threshold exceeded
- **Dynamic HDC_DIM**: Runtime configurable (512, 1024, 2048, 4096, etc.) via `hv_dim` parameter

### GPU Support

- **Inference**: `use_gpu=True` enables CUDA batch prediction
- **Training**: `use_gpu_training=True` enables parallel AdaptHD on GPU with capability-based kernel selection
- **Capability-based selection**: `ampere` (sm_80+: cooperative kernel, 100KB shared) / `turing` (sm_70-75: 64KB shared) / `legacy` (sm_61 and below)
- **Fallback**: If CUDA unavailable, automatically falls back to CPU (AVX2/NEON/SCALAR)

### CUDA 2026 Implementation

```
src/
├── addernet_hdc_train_cuda.cu        # Generic AdaptHD retrain kernel (nvcc)
└── cuda_train/
    └── addernet_hdc_train_cuda_2026.cu  # Cooperative kernel for Ampere+
```

- **Basic CUDA retrain** (`addernet_hdc_train_cuda.cu`): Fills the missing `an_hdc_retrain_cuda` symbol. Pre-encodes samples on CPU, uploads to GPU for Hamming search and count updates via `atomicAdd`.
- **Cooperative 2026 kernel** (`addernet_hdc_train_cuda_2026.cu`): Ampere-optimized with shared memory tiering (48KB/64KB/100KB), warp-level operations, and persistent kernel mode. Built by `make cuda_2026`.
- **Kernel selection** (Phase 2): `addernet_hdc.py` uses `_kernel_variant` from `cuda_detector` to route to the optimal kernel: 2026 cooperative → generic CUDA → CPU.

## Testing

### C Tests (`tests/`)

- `test_main.c` — Tests AdderNetLayer (Celsius→Fahrenheit conversion, save/load, batch prediction)
- `test_hdc_main.c` — Tests AdderNetHDC (Iris dataset, encoding, prediction)

### Python Tests

- `test_validation.py` — Comprehensive validation against scikit-learn datasets
- `test_attention.py` — Attention mechanism tests

## Important Constraints

- **Input values**: Cast to `int` internally (fractional parts truncated). Use scaling/range mapping
- **Table size**: Capped at 256 entries (`AN_TABLE_SIZE` constant)
- **Single-variable only**: Each AdderNetLayer handles one input variable
- **No versioning in save format**: Binary format is `[int size][int bias][int input_min][int input_max][double lr][double offsets...]`

## Platform-Specific Notes

- **Linux**: `.so` files, Makefile uses `-fPIC -shared`
- **macOS**: `.dylib` files (handled in `__init__.py`)
- **Windows**: `.dll` files (handled in `__init__.py`)
- **x86_64**: Uses `-mavx2 -mpopcnt -march=native`
- **ARM**: Uses `-march=armv8-a+simd -mfpu=neon`
- **CUDA**: `make cuda` (inline PTX via gcc, no nvcc) vs `make cuda_native` (requires nvcc, old kernel) vs `make cuda_2026` (Ampere+ cooperative kernel)

## Version Notes

- **pyproject.toml** declares version `1.3.8`
- **setup.py** also uses version `1.3.8`
- **`addernet/__init__.py`** still has `__version__ = "1.2.7"` — this is stale and should be updated

## Dependencies

- **Runtime**: `numpy`, `scikit-learn`, `scipy` (from pyproject.toml)
- **Dev**: `pytest` (optional)
- **C build**: `gcc`, `make`; `nvcc` optional for native CUDA
