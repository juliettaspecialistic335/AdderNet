/*
 * AdderNet Library — Public API
 * ==============================
 *   Neural network using pure addition instead of multiplication.
 *   Inference:  result = tab[(input + bias) & mask]   (1 load, zero arithmetic)
 *
 *   Compile: gcc -O3 -march=native -fPIC -shared -o libaddernet.so addernet.c -lm
 */

#ifndef ADDERNET_H
#define ADDERNET_H

#include <stdalign.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Safe aligned_alloc: pads size to multiple of alignment (C11 compliance) */
static inline void *an_aligned_alloc(size_t alignment, size_t size) {
    size_t padded = (size + alignment - 1) & ~(alignment - 1);
    return aligned_alloc(alignment, padded);
}

/*
 * Table must be power-of-2 sized for AND-mask indexing.
 * 256 entries covers [-50, 205] with bias=50.
 */
#define AN_TABLE_BITS  8
#define AN_TABLE_SIZE  (1 << AN_TABLE_BITS)   /* 256 */
#define AN_TABLE_MASK  (AN_TABLE_SIZE - 1)    /* 0xFF */

/*
 * An AdderNet layer.
 *   offset[i] = precomputed output for index i
 *   Index:   idx = (int_input + bias) & mask
 *   Predict: result = offset[idx]
 *
 * The offset table is aligned to 64 bytes (cache line).
 */
typedef struct {
    int    size;                    /* table size (must be power of 2) */
    int    bias;                    /* added to input before masking  */
    int    input_min;               /* min training input value        */
    int    input_max;               /* max training input value        */
    double lr;                      /* learning rate                   */
    alignas(64) double offset[AN_TABLE_SIZE];
} an_layer;

/* ==================================================================
 *  API — 7 functions
 * ================================================================== */

/*
 * an_layer_create — Allocate and initialize a layer.
 *
 *   size:      table size (must be power of 2, e.g. 256)
 *   bias:      offset added to input before table index (e.g. 50)
 *   input_min: minimum expected input value (for data expansion)
 *   input_max: maximum expected input value (for data expansion)
 *   lr:        learning rate (e.g. 0.1)
 *
 *   Returns: pointer to new layer, or NULL on failure.
 *   Caller must free with an_layer_free().
 */
an_layer *an_layer_create(int size, int bias, int input_min, int input_max, double lr);

/*
 * an_layer_free — Free a layer allocated by an_layer_create().
 */
void an_layer_free(an_layer *layer);

/*
 * an_train — Train the layer on input→target pairs.
 *
 *   Phase 1: train on raw samples (epochs_raw iterations).
 *   Phase 2: expand data via interpolation + extrapolation,
 *            then train again (epochs_expanded iterations).
 *
 *   inputs:  array of n_samples input values (doubles, truncated to int for indexing)
 *   targets: array of n_samples target values
 *   Returns: 0 on success, -1 on error.
 */
int an_train(an_layer *layer,
             const double *inputs, const double *targets,
             int n_samples, int epochs_raw, int epochs_expanded);

/*
 * an_predict — Single input inference.
 *
 *   result = layer->offset[(int(input) + bias) & mask]
 */
double an_predict(const an_layer *layer, double input);

/*
 * an_predict_batch — Batch inference on an array.
 *
 *   inputs:  array of n input values
 *   outputs: array of n output values (pre-allocated by caller)
 *   Returns: 0 on success, -1 on error.
 */
int an_predict_batch(const an_layer *layer,
                     const double *inputs, double *outputs, int n);

/*
 * an_save — Serialize layer to a binary file.
 *
 *   Format: [int size][int bias][int input_min][int input_max][double lr]
 *           [double offset[0]..offset[size-1]]
 *
 *   Returns: 0 on success, -1 on error.
 */
int an_save(const an_layer *layer, const char *path);

/*
 * an_load — Deserialize layer from a binary file.
 *
 *   Returns: pointer to new layer, or NULL on failure.
 *   Caller must free with an_layer_free().
 */
an_layer *an_load(const char *path);

/*
 * an_get_offset — Copy the offset table into caller's buffer.
 *
 *   buf: must have room for at least layer->size doubles.
 *   Returns: 0 on success, -1 on error.
 */
int an_get_offset(const an_layer *layer, double *buf, int buf_size);

/* Metadata accessors (for opaque pointer usage from FFI) */
int    an_get_size(const an_layer *layer);
int    an_get_bias(const an_layer *layer);
int    an_get_input_min(const an_layer *layer);
int    an_get_input_max(const an_layer *layer);
double an_get_lr(const an_layer *layer);

/*
 * an_layer_mmap_load — Memory-map a saved LUT file for zero-copy loading.
 *
 *   Opens the file read-only and maps it with MAP_SHARED.
 *   The OS manages paging — no full copy to RAM.
 *   Returns: pointer to mapped data, or NULL on failure.
 *   Linux only. Caller should munmap() when done.
 */
void *an_layer_mmap_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* ADDERNET_H */
