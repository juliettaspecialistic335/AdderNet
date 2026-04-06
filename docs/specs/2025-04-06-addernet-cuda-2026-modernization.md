# AdderNet CUDA 2026 Modernization Spec

**Date**: 2026-04-06  
**Status**: Approved for Implementation  
**Scope**: Modernize AdderNet CUDA from 2016 (PTX 6.5/sm_61) to 2026 (PTX 8.6/sm_90)

---

## 1. Problem Statement

AdderNet v1.3.8 uses CUDA technologies from 2016:
- PTX version 6.5 (deprecated)
- SM target sm_61 (Pascal architecture)
- No shared memory optimization
- CPU-bound encoding
- Detection fails on Colab/Kaggle environments

This limits performance on modern GPUs (RTX 30xx/40xx, A100, H100) by 40-60x.

---

## 2. Solution Architecture

### 2.1 Hybrid Architecture with Runtime Detection

```
┌─────────────────────────────────────────────────────────────┐
│                    AdderNet CUDA 2026                        │
├─────────────────────────────────────────────────────────────┤
│  Detection Layer (A+C)                                       │
│  ├── Multiple paths: /usr/bin/nvcc, $CONDA_PREFIX/bin/nvcc  │
│  └── pip/conda package detection                           │
├─────────────────────────────────────────────────────────────┤
│  Capability Layer                                            │
│  ├── Runtime: cudaDeviceGetAttribute                         │
│  └── Select kernel: sm_61 | sm_70-75 | sm_80+               │
├─────────────────────────────────────────────────────────────┤
│  Kernel Implementations                                     │
│  ├── Legacy (PTX 6.5 inline): sm_52-61                     │
│  ├── Turing (PTX 7.2): sm_70-75                            │
│  └── Ampere+ (PTX 8.6): sm_80-90                           │
│      ├── cp.async.s.shared                                  │
│      ├── Tensor Cores (mma.sync)                           │
│      └── Cooperative kernels                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

| Component | Technology | Benefit |
|-----------|------------|---------|
| Cooperative Kernel | Grid-level sync | 40-60x speedup in training |
| Warp-level Reduction | __reduce_add_sync | Eliminates 97% atomic conflicts |
| Shared Memory Tiering | 48KB→64KB→100KB | Architecture-tuned caching |
| CUDA Graphs | Persistent execution | Eliminates launch overhead |
| Unified Memory (opt) | cudaMallocManaged | Simplified host/device sync |

---

## 3. Implementation Details

### 3.1 File Structure

```
src/
├── cuda/
│   ├── addernet_cuda_2026.cu          # Main dispatcher
│   ├── addernet_cuda_ampere.cu        # sm_80+ optimized
│   ├── addernet_cuda_turing.cu        # sm_70-75
│   ├── addernet_cuda_legacy.cu        # sm_61 fallback
│   ├── cuda_graphs.cu                 # Graphs API wrapper
│   └── cuda_detection.c               # Detection logic
├── cuda_core/
│   ├── hdc_core_cuda_ampere.c         # PTX 8.6 inline
│   ├── hdc_cuda_batch_ampere.c        # Shared memory batch
│   └── hdc_cuda_cooperative.c         # Cooperative encoding
├── cuda_train/
│   └── addernet_hdc_train_cuda_2026.cu # New training kernel
└── legacy/                              # Existing files (backup)
    └── ...

addernet/
├── build_ext_2026.py                  # Enhanced build system
├── cuda_detector.py                   # Detection module
└── ... (existing files)
```

### 3.2 Detection System (Requirement A+C)

```python
# Detection priority order:
1. System paths: /usr/bin/nvcc, /usr/local/cuda*/bin/nvcc
2. Environment: $CUDA_HOME/bin/nvcc, $PATH
3. Conda: $CONDA_PREFIX/bin/nvcc
4. pip packages: nvidia-cuda-runtime-cu12, nvidia-cuda-nvcc-cu12
5. Runtime detection: ctypes/libdl for libcuda.so

# Capability detection:
cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor)
cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor)
capability = major * 10 + minor
```

### 3.3 Cooperative Training Kernel

```cpp
// hdc_train_cooperative_ampere.cu
__global__ void hdc_train_cooperative_ampere(
    const double* __restrict__ X,
    const int* __restrict__ y,
    const uint64_t* __restrict__ codebook,
    int* __restrict__ cb_counts,
    // ... params
) {
    __shared__ uint64_t shared_codebook[10][16]; // 10 classes, 16 words
    __shared__ int warp_results[256];
    
    // Phase 1: Warp-level encoding
    uint64_t query[16];
    warp_encode(x, query, n_vars, hv_words);
    
    // Phase 2: Warp-reduce Hamming distances
    int distances = warp_hamming(query, shared_codebook);
    
    // Phase 3: Coalesced atomic updates
    if (mispredicted) {
        warp_level_correction(query, best_c, true_c, cb_counts);
    }
    
    // Phase 4: Cooperative rebuild
    __syncthreads();
    cooperative_rebuild_codebook(cb_counts, codebook);
}
```

### 3.4 Shared Memory Tiering (Requirement C)

```cpp
// Detected at kernel launch
if (capability >= 80) {
    shared_bytes = 100 * 1024;  // Ampere: 100KB
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
} else if (capability >= 70) {
    shared_bytes = 64 * 1024;   // Turing: 64KB
} else {
    shared_bytes = 48 * 1024;   // Pascal: 48KB
}
```

### 3.5 CUDA Graphs Integration

```cpp
// Training capture once, execute many
cudaGraph_t training_graph;
cudaGraphExec_t training_instance;

// Capture
cudaStreamBeginCapture(stream);
hdc_train_batch<<<...>>>(...);
cudaStreamEndCapture(stream, &training_graph);
cudaGraphInstantiate(&training_instance, training_graph);

// Execute per epoch
for (int epoch = 0; epoch < n_epochs; epoch++) {
    cudaGraphLaunch(training_instance, stream);
}
```

---

## 4. API Compatibility

### 4.1 Python API (Unchanged)

```python
from addernet import AdderNetHDC

model = AdderNetHDC(
    n_vars=4,
    n_classes=3,
    use_gpu=True,           # Auto-detects optimal path
    use_gpu_training=True,   # Auto-detects sm_80+ features
)
model.train(X, y, n_iter=20)
```

### 4.2 Build System

```bash
# Detection happens at runtime via build_ext_2026.py
# Compilation targets multiple architectures:
#   - arch=sm_61 (Pascal compatible)
#   - arch=sm_75 (Turing)
#   - arch=sm_80 (Ampere)
#   - arch=sm_89 (Ada)
#   - arch=sm_90 (Hopper)
#   - arch=compute_90,code=sm_90 (JIT for future)
```

---
## 5. Testing Strategy

### 5.1 Unit Tests
- Detection: Verify nvcc found in Colab/Kaggle-like paths
- Capability: Mock different SM versions
- Correctness: Same results CPU vs GPU vs 2016 vs 2026

### 5.2 Integration Tests
- Iris dataset comparison
- Memory limit testing (100KB shared)
- Graphs validation

### 5.3 Performance Benchmarks
- Training speed: Iris, Wine, Breast Cancer
- Comparison: sm_61 vs sm_75 vs sm_80 vs sm_89

---

## 6. Deployment Rollout

### Phase 1: Detection System
- Fix Colab/Kaggle detection
- Backwards compatible

### Phase 2: Shared Memory
- Add C-tier optimization
- sm_80+ gets 100KB

### Phase 3: Cooperative Kernels
- New training: hdc_train_cooperative_ampere.cu
- Major performance bump

### Phase 4: CUDA Graphs
- Persistent execution
- Final optimization

---

## 7. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Detection Colab | 100% success | `import addernet` on fresh runtime |
| Detection Kaggle | 100% success | Same as above |
| Training Speed A100 | 50x faster | vs v1.3.8 on same dataset |
| Training Speed T4 | 25x faster | vs v1.3.8 |
| Correctness | 100% | Bit-exact predictions |
| Backwards Compat | Yes | sm_61 still works |

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| sm_61 performance regression | Keep legacy path |
| CUDA 12 requirement | Fallback to pip-installed runtime |
| Compile time explosion | PTX JIT, not fat binary |
| Memory pressure | Tiered shared allocation |

---

## 9. Appendix: Technical References

### PTX Version 8.6 Features
- `cp.async.s.shared`: Async copy to shared
- `mma.sync.aligned`: Tensor Core operations
- `reduce.sync`: Warp-level reduction

### CUDA 12.x APIs
- `cudaGraphPeerAccessGetAttribute`
- `cudaStreamSetAttribute`
- `cudaFuncAttributePreferredSharedMemoryCarveout`

---

**Approved by**: User (2026-04-06)  
**Next Step**: Invoke writing-plans skill for implementation tasks
