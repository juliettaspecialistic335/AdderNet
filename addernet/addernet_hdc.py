#!/usr/bin/env python3
"""
AdderNet-HDC Python Bindings — ctypes interface to libaddernet_hdc.so
======================================================================

Usage:
    from addernet_hdc import AdderNetHDC

    model = AdderNetHDC(n_vars=4, n_classes=3, table_size=256)
    model.train(X, y)
    pred = model.predict(x)
"""

import os
import ctypes
import numpy as np

# ---- Locate shared library ----

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.join(_HERE, "..", "build")
_LIB_NAMES = [
    os.path.join(_BUILD, "libaddernet_hdc.so"),
    os.path.join(_HERE, "libaddernet_hdc.so"),
    os.path.join(_HERE, "libaddernet_hdc.dylib"),
    "libaddernet_hdc.so",
]

_lib = None
for _name in _LIB_NAMES:
    try:
        _lib = ctypes.CDLL(_name)
        break
    except OSError:
        continue

if _lib is None:
    raise OSError(
        "Cannot find libaddernet_hdc.so. "
        "Build it first: cd addernet_lib && make hdc"
    )

# ---- Opaque pointer type ----

_AnHdcPtr = ctypes.c_void_p

# ---- Function signatures ----

_lib.an_hdc_create.restype = _AnHdcPtr
_lib.an_hdc_create.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.an_hdc_free.restype = None
_lib.an_hdc_free.argtypes = [_AnHdcPtr]

_lib.an_hdc_train.restype = None
_lib.an_hdc_train.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

_lib.an_hdc_retrain.restype = ctypes.c_int
_lib.an_hdc_retrain.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]

_lib.an_hdc_predict.restype = ctypes.c_int
_lib.an_hdc_predict.argtypes = [_AnHdcPtr, ctypes.POINTER(ctypes.c_double)]

_lib.an_hdc_predict_batch.restype = ctypes.c_int
_lib.an_hdc_predict_batch.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

# Melhoria 4: AVX2 batch prediction
_lib.an_hdc_predict_batch_avx.restype = ctypes.c_int
_lib.an_hdc_predict_batch_avx.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

if _lib_cuda is not None:
    _lib_cuda.an_hdc_predict_batch_cuda.restype = ctypes.c_int
    _lib_cuda.an_hdc_predict_batch_cuda.argtypes = [
        _AnHdcPtr,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]

_lib.an_hdc_save.restype = ctypes.c_int
_lib.an_hdc_save.argtypes = [_AnHdcPtr, ctypes.c_char_p]

_lib.an_hdc_load.restype = _AnHdcPtr
_lib.an_hdc_load.argtypes = [ctypes.c_char_p]

# OPT-1: Cache functions
_lib.an_hdc_warm_cache.restype = None
_lib.an_hdc_warm_cache.argtypes = [_AnHdcPtr]

_lib.an_hdc_set_cache.restype = None
_lib.an_hdc_set_cache.argtypes = [_AnHdcPtr, ctypes.c_int]

# OPT-5: Multithreaded batch prediction
_lib.an_hdc_predict_batch_mt.restype = ctypes.c_int
_lib.an_hdc_predict_batch_mt.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]

_lib.an_hdc_set_threads.restype = None
_lib.an_hdc_set_threads.argtypes = [_AnHdcPtr, ctypes.c_int]

# Melhoria 3: Hadamard encoding
_lib.an_hdc_set_hadamard.restype = None
_lib.an_hdc_set_hadamard.argtypes = [_AnHdcPtr, ctypes.c_int]

# LSH: Locality-Sensitive Hashing
_lib.an_hdc_build_lsh.restype = None
_lib.an_hdc_build_lsh.argtypes = [_AnHdcPtr]

_lib.an_hdc_build_lsh_ex.restype = None
_lib.an_hdc_build_lsh_ex.argtypes = [_AnHdcPtr, ctypes.c_int, ctypes.c_int]

_lib.an_hdc_set_lsh.restype = None
_lib.an_hdc_set_lsh.argtypes = [_AnHdcPtr, ctypes.c_int]

# predict_top_k
_lib.an_hdc_predict_top_k.restype = None
_lib.an_hdc_predict_top_k.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

# Problem 6: Interaction encoding
_lib.an_hdc_set_interactions.restype = None
_lib.an_hdc_set_interactions.argtypes = [
    _AnHdcPtr,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

# OPT-8: Backend detection
_lib.hdc_detect_backend.restype = ctypes.c_int
_lib.hdc_detect_backend.argtypes = []

_lib.hv_seed.restype = None
_lib.hv_seed.argtypes = [ctypes.c_uint]

# HDC primitive bindings (for direct hypervector manipulation)

_HV_WORDS = 40  # HDC_WORDS = ceil(2500/64)
_HV_DIM = 2500  # HDC_DIM — actual hypervector dimensionality
HDC_BYTES = _HV_WORDS * 8  # 320 bytes per hypervector
_HV_t = ctypes.c_uint64 * _HV_WORDS

_lib.hv_bind.restype = None
_lib.hv_bind.argtypes = [_HV_t, _HV_t, _HV_t]

_lib.hv_bundle.restype = None
_lib.hv_bundle.argtypes = [_HV_t, ctypes.POINTER(_HV_t), ctypes.c_int]

_lib.hv_hamming.restype = ctypes.c_int
_lib.hv_hamming.argtypes = [_HV_t, _HV_t]

_lib.hv_similarity.restype = ctypes.c_float
_lib.hv_similarity.argtypes = [_HV_t, _HV_t]

_lib.hv_random.restype = None
_lib.hv_random.argtypes = [_HV_t]

_lib.hv_add_noise.restype = None
_lib.hv_add_noise.argtypes = [_HV_t, _HV_t, ctypes.c_float]

_lib.hv_copy.restype = None
_lib.hv_copy.argtypes = [_HV_t, _HV_t]

_lib.hv_add_noise.restype = None
_lib.hv_add_noise.argtypes = [_HV_t, _HV_t, ctypes.c_float]


# ---- Python wrapper ----

class AdderNetHDC:
    """
    Multivariate classifier using AdderNet encoding + Hyperdimensional Computing.
    Zero floating-point multiplication at inference.
    """

    def __init__(self, n_vars=1, n_classes=2, table_size=256, bias=None,
                 seed=42, _ptr=None):
        """
        Create a new model.

        Args:
            n_vars:     number of input variables
            n_classes:  number of output classes
            table_size: encoding table size per variable (power of 2)
            bias:       list of bias values per variable (default: table_size//2)
            seed:       random seed for reproducibility
        """
        if _ptr is not None:
            self._ptr = _ptr
            self._n_vars = n_vars
            self._n_classes = n_classes
            self._table_size = table_size
            return

        _lib.hv_seed(seed)

        bias_arr = None
        if bias is not None:
            bias_arr = (ctypes.c_int * n_vars)(*bias)

        self._ptr = _lib.an_hdc_create(n_vars, n_classes, table_size, bias_arr)
        if not self._ptr:
            raise MemoryError("an_hdc_create failed")
        self._n_vars = n_vars
        self._n_classes = n_classes
        self._table_size = table_size

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.an_hdc_free(self._ptr)
            self._ptr = None

    def train(self, X, y, n_iter=0, lr=1.0, margin=0, regenerate=0.0,
              patience=10, verbose=False, interactions=0):
        """
        Train the codebook from labeled data, with optional iterative retraining.

        Uses OnlineHD weighted bundling by default to prevent model saturation.
        When n_iter > 0, applies RefineHD iterative correction with additive margin
        and NeuralHD dimension regeneration.

        Args:
            X: n_samples × n_vars array (list of lists or numpy array)
            y: n_samples class labels (int, 0-indexed)
            n_iter: number of iterative correction passes (0 = single-pass, default)
            lr: learning rate for iterative retraining (default 1.0)
            margin: RefineHD margin — aceita 4 formas:
                    0 / None  → desligado (AdaptHD puro)
                    float 0-1 → fração de D (ex: 0.05 = 5% de D)
                    '5%'      → percentual string (ex: '5%', '10%')
                    int > 0   → distância Hamming absoluta em bits
            regenerate: NeuralHD dimension regeneration rate (0.0 = off, 0.02-0.05 recommended)
            patience: early stopping patience in epochs (0 = disabled, 5 recommended)
            verbose: if True, print epoch progress to stderr every 10 epochs. If int, print every N epochs.
            interactions: number of top correlated feature pairs to encode (0 = disabled, 10 recommended)
        Returns:
            dict with training history:
                'epochs_run': epochs actually executed,
                'best_val_accuracy': best validation accuracy (last 25% of data),
                'best_train_accuracy': best training accuracy (first 75% of data),
                'best_epoch': epoch of best val accuracy,
                'stopped_early': True if early stopping triggered
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int32)

        if X.ndim == 1:
            X = X.reshape(-1, self._n_vars)

        n = X.shape[0]
        if len(y) != n:
            raise ValueError(f"X has {n} samples but y has {len(y)}")

        # Default history for single-pass training
        history = {
            'epochs_run': 0,
            'best_val_accuracy': 0.0,
            'best_train_accuracy': 0.0,
            'best_epoch': 0,
            'stopped_early': False,
        }

        # Problema 6: Detect and set interaction pairs
        if interactions > 0 and X.shape[1] > 1:
            corr = np.corrcoef(X.T)
            pairs = []
            for ii in range(corr.shape[0]):
                for jj in range(ii + 1, corr.shape[1]):
                    pairs.append((abs(corr[ii, jj]), ii, jj))
            pairs.sort(reverse=True)
            top_pairs = pairs[:interactions]
            if top_pairs:
                pairs_i = np.array([p[1] for p in top_pairs], dtype=np.int32)
                pairs_j = np.array([p[2] for p in top_pairs], dtype=np.int32)
                _lib.an_hdc_set_interactions(
                    self._ptr,
                    pairs_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    pairs_j.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    len(top_pairs),
                )

        # Initial train (encodes samples including interactions)
        _lib.an_hdc_train(
            self._ptr,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n,
        )

        if n_iter > 0:
            D = _HV_WORDS * 64  # total bits no hipervector (2500)

            # ── Conversão de margin ──────────────────────────────
            if margin == 0 or margin is None:
                margin_int = 0          # desligado — AdaptHD puro

            elif isinstance(margin, str) and margin.endswith('%'):
                pct = float(margin[:-1]) / 100.0
                margin_int = max(1, int(pct * D))

            elif isinstance(margin, float) and 0.0 < margin < 1.0:
                margin_int = max(1, int(margin * D))

            elif isinstance(margin, int) and margin > 0:
                margin_int = margin

            else:
                raise ValueError(
                    f"margin deve ser: 0 (off), float 0-1 (fração de D), "
                    f"'5%' (percentual), ou int (bits). Recebeu: {margin!r}"
                )

            # Clamp: nunca maior que 20% de D
            margin_int = min(margin_int, int(D * 0.20))

            if verbose:
                if margin_int == 0:
                    print(f"  margin=off (AdaptHD puro)")
                else:
                    print(f"  margin={margin!r} → {margin_int} bits ({margin_int/D*100:.1f}% de D={D})")

            # Verbose level: True → 10, int → as-is, False → 0
            if verbose is True:
                verbose_level = 10
            elif isinstance(verbose, int) and verbose > 0:
                verbose_level = verbose
            else:
                verbose_level = 0

            epochs_run = ctypes.c_int(0)
            _lib.an_hdc_retrain(
                self._ptr,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                n,
                n_iter,
                ctypes.c_float(lr),
                ctypes.c_int(margin_int),
                ctypes.c_float(regenerate),
                ctypes.c_int(patience),
                ctypes.c_int(verbose_level),
                ctypes.byref(epochs_run),
            )

            # Compute final accuracies on train/val split
            n_val = max(1, n // 4)
            n_train_split = n - n_val
            train_acc = float(self.accuracy(X[:n_train_split], y[:n_train_split]))
            val_acc = float(self.accuracy(X[n_train_split:], y[n_train_split:]))

            history['epochs_run'] = epochs_run.value
            history['best_val_accuracy'] = val_acc
            history['best_train_accuracy'] = train_acc
            history['best_epoch'] = epochs_run.value
            history['stopped_early'] = (epochs_run.value < n_iter)

        return history

    def predict(self, x):
        """
        Classify one sample.

        Args:
            x: list or array of n_vars input values
        Returns:
            predicted class label (int)
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        return _lib.an_hdc_predict(
            self._ptr,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    def predict_batch(self, X):
        """
        Classify multiple samples.

        Args:
            X: n_samples × n_vars array
        Returns:
            numpy array of predicted class labels
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, self._n_vars)
        n = X.shape[0]
        outputs = np.empty(n, dtype=np.int32)
        
        if getattr(self, 'use_gpu', False):
            if _lib_cuda is None:
                raise RuntimeError("CUDA backend requested but libaddernet_cuda.so not found")
            _lib_cuda.an_hdc_predict_batch_cuda(
                self._ptr,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                n,
            )
        else:
            _lib.an_hdc_predict_batch(
                self._ptr,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                n,
            )
        return outputs

    def predict_batch_avx(self, X):
        """
        Classify multiple samples using AVX2 SIMD (Melhoria 4).
        Processes 4 samples simultaneously for faster batch inference.

        Args:
            X: n_samples × n_vars array
        Returns:
            numpy array of predicted class labels
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, self._n_vars)
        n = X.shape[0]
        outputs = np.empty(n, dtype=np.int32)
        _lib.an_hdc_predict_batch_avx(
            self._ptr,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n,
        )
        return outputs

    def predict_batch_mt(self, X, n_threads=0):
        """
        Classify multiple samples using multiple threads (OPT-5).

        Args:
            X: n_samples × n_vars array
            n_threads: number of threads (0 = auto-detect)
        Returns:
            numpy array of predicted class labels
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, self._n_vars)
        n = X.shape[0]
        outputs = np.empty(n, dtype=np.int32)
        _lib.an_hdc_predict_batch_mt(
            self._ptr,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n,
            n_threads,
        )
        return outputs

    def warm_cache(self):
        """Pre-compute encoding cache (OPT-1). Call once before benchmarking."""
        _lib.an_hdc_warm_cache(self._ptr)

    def set_cache(self, use_cache):
        """Enable/disable encoding cache (OPT-1)."""
        _lib.an_hdc_set_cache(self._ptr, 1 if use_cache else 0)

    def set_threads(self, n_threads):
        """Set thread count for batch prediction (OPT-5)."""
        _lib.an_hdc_set_threads(self._ptr, n_threads)

    def set_hadamard(self, enable=True):
        """Enable/disable Hadamard encoding (Melhoria 3 - orthogonal base vectors)."""
        _lib.an_hdc_set_hadamard(self._ptr, 1 if enable else 0)

    def build_lsh(self, k=10, l=8):
        """Build LSH index with K=10, L=8 tables."""
        _lib.an_hdc_build_lsh_ex(self._ptr, k, l)

    def set_lsh(self, enable):
        """Enable/disable LSH for prediction."""
        _lib.an_hdc_set_lsh(self._ptr, 1 if enable else 0)

    def predict_top_k(self, x, k=5):
        """
        Classify one sample and return top K predictions.

        Args:
            x: list or array of n_vars input values
            k: number of top predictions to return
        Returns:
            list of top K predicted class labels (int)
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        out_classes = np.empty(k, dtype=np.int32)
        _lib.an_hdc_predict_top_k(
            self._ptr,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_classes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            k,
        )
        return out_classes.tolist()

    def accuracy(self, X, y):
        """
        Compute accuracy on given data.

        Args:
            X: n_samples × n_vars array
            y: true class labels
        Returns:
            accuracy as float
        """
        y_pred = self.predict_batch(X)
        y = np.ascontiguousarray(y, dtype=np.int32)
        return np.mean(y_pred == y)

    def add_noise(self, hv, temperature):
        """
        Add noise to a hypervector with given temperature.

        Args:
            hv: hypervector (numpy array of uint64)
            temperature: 0.0 = no noise, 1.0 = total noise
        Returns:
            new hypervector with noise applied
        """
        import random
        src = np.ascontiguousarray(hv, dtype=np.uint64).copy()
        dst = src.copy()
        
        if temperature <= 0.0:
            return dst
        if temperature >= 1.0:
            for w in range(_HV_WORDS):
                for b in range(64):
                    if random.random() < temperature:
                        dst[w] ^= (1 << b)
            return dst
        
        threshold = int(temperature * 256)
        for w in range(_HV_WORDS):
            for b in range(64):
                if random.randint(0, 255) < threshold:
                    dst[w] ^= (1 << b)
        return dst

    def bundle_classes(self, class_indices):
        """
        Bundle multiple class prototypes into one (criatividade).

        Args:
            class_indices: list of class indices to bundle
        Returns:
            bundled hypervector
        """
        cb = self.codebook
        n = len(class_indices)
        result = np.zeros(_HV_WORDS, dtype=np.uint64)
        
        if n == 0:
            return result
        if n == 1:
            return cb[class_indices[0]].copy()
        
        for i, idx in enumerate(class_indices):
            if i == 0:
                result[:] = cb[idx][:]
            else:
                counts = np.zeros(_HV_WORDS * 64, dtype=np.uint16)
                threshold = (i + 1) // 2
                for w in range(_HV_WORDS):
                    for bit in range(64):
                        if (cb[idx][w] >> bit) & 1:
                            counts[w * 64 + bit] += 1
                for w in range(_HV_WORDS):
                    for bit in range(64):
                        if counts[w * 64 + bit] > threshold:
                            result[w] |= (1 << bit)
        return result

    def classify_hv(self, hv):
        """
        Classify a hypervector by finding closest codebook entry.

        Args:
            hv: hypervector to classify (numpy array or list)
        Returns:
            predicted class index
        """
        cb = self.codebook
        hv_arr = np.ascontiguousarray(hv, dtype=np.uint64)
        best_c = 0
        best_d = int(np.sum(hv_arr != cb[0]))
        for c in range(1, self._n_classes):
            d = int(np.sum(hv_arr != cb[c]))
            if d < best_d:
                best_d = d
                best_c = c
        return best_c

    def save(self, path):
        """Save model to binary file."""
        ret = _lib.an_hdc_save(self._ptr, path.encode("utf-8"))
        if ret != 0:
            raise IOError(f"an_hdc_save failed: {path}")

    @classmethod
    def load(cls, path):
        """Load model from binary file."""
        ptr = _lib.an_hdc_load(path.encode("utf-8"))
        if not ptr:
            raise IOError(f"an_hdc_load failed: {path}")
        return cls(_ptr=ptr)

    @property
    def n_vars(self):
        return self._n_vars

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def table_size(self):
        return self._table_size

    @property
    def codebook(self):
        """
        Return the trained codebook as a list of numpy uint64 arrays.
        Each array has 157 words (10000 bits).

        Requires the model to be trained first.
        """
        _HDC_WORDS = _HV_WORDS

        # Read the codebook pointer from the an_hdc_model struct.
        # On x86_64, the codebook hv_t* field is at byte offset 32
        # (after n_vars, n_classes, table_size, table_mask, bias*, enc_table*).
        buf = ctypes.string_at(self._ptr, 48)  # read first 48 bytes of struct
        cb_addr = int.from_bytes(buf[32:40], byteorder='little')
        if cb_addr == 0:
            raise RuntimeError("Model not trained yet (codebook is NULL)")

        # Cast the address to a flat uint64 array
        cb = ctypes.cast(
            ctypes.c_void_p(cb_addr),
            ctypes.POINTER(ctypes.c_uint64 * (_HDC_WORDS * self._n_classes))
        )

        result = []
        flat = cb.contents  # the full (HDC_WORDS * n_classes) uint64 array
        for c in range(self._n_classes):
            offset = c * _HDC_WORDS
            hv = np.array([flat[offset + i] for i in range(_HDC_WORDS)],
                          dtype=np.uint64)
            result.append(hv)
        return result

    def __repr__(self):
        return (f"AdderNetHDC(n_vars={self._n_vars}, "
                f"n_classes={self._n_classes}, "
                f"table_size={self._table_size})")


def hdc_detect_backend():
    """
    Detect the available backend for HDC operations.

    Returns:
        'AVX2', 'NEON', or 'SCALAR'
    """
    backends = ["SCALAR", "AVX2", "NEON"]
    return backends[_lib.hdc_detect_backend()]
ackends[_lib.hdc_detect_backend()]
