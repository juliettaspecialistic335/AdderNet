#ifndef HDC_CORE_H
#define HDC_CORE_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Safe aligned_alloc wrapper: pads size to multiple of alignment.
 * C11 requires size to be a multiple of alignment, otherwise returns NULL.
 * This wraps it to always round up, preventing silent NULL returns. */
static inline void *safe_aligned_alloc(size_t alignment, size_t size) {
    size_t padded = (size + alignment - 1) & ~(alignment - 1);
    return aligned_alloc(alignment, padded);
}

typedef uint64_t* hv_t;
typedef const uint64_t* const_hv_t;

/* Generate a random hypervector (approximately 50% bits set) */
void hv_random(hv_t out, int hv_words, int hv_dim);

/* Seed the internal RNG (for reproducibility) */
void hv_seed(unsigned int seed);
uint64_t hv_seed_state(void);
void hv_seed_set(uint64_t state);

/* out = a XOR b */
void hv_bind(hv_t out, const_hv_t a, const_hv_t b, int hv_words);

/* out = majority vote of n vectors (bundling) */
void hv_bundle(hv_t out, const_hv_t *vecs, int n, int hv_words, int hv_dim);
void hv_bundle_flat(hv_t out, const uint64_t *flat_vecs, int n, int hv_words, int hv_dim);
void hv_bundle_weighted(hv_t out, const_hv_t *vecs, const int *weights, int n, int hv_words, int hv_dim);

/* Hamming distance: number of differing bits */
int hv_hamming(const_hv_t a, const_hv_t b, int hv_words);
int hv_hamming_unrolled(const_hv_t a, const_hv_t b, int hv_words);
void hv_hamming_batch4(const uint64_t *queries, const uint64_t *ref, int *dists, int hv_words);

/* Cosine/Hamming similarity [0.0, 1.0] */
float hv_similarity(const_hv_t a, const_hv_t b, int hv_words, int hv_dim);

/* Copy/Zero */
void hv_copy(hv_t dst, const_hv_t src, int hv_words);
void hv_zero(hv_t dst, int hv_words);

/* Generate deterministic HV from seed (no memory required) */
void hv_from_seed(hv_t out, uint64_t seed, int hv_words);
void hv_from_seed_fast(hv_t out, uint64_t seed, int hv_words, int hv_dim);
void hv_from_hadamard(hv_t out, int var, int bin, int hv_words, int hv_dim);

/* Flip random bits with given probability (temperature) */
void hv_add_noise(hv_t out, const_hv_t src, float temperature, int hv_words);

/* ---- Circulant Encoding ---- */
typedef struct {
    uint64_t  *base_vector;
    int        circulant_ready;
} hdc_circulant_t;

void hdc_circulant_init(hdc_circulant_t *circ, uint64_t seed_val, uint64_t seed_pos, int hv_words, int hv_dim);
void hv_rotate(hv_t out, const_hv_t src, int shift, int hv_words, int hv_dim);
void hv_from_circulant(hv_t out, const hdc_circulant_t *circ, int var, int bin, int hv_words, int hv_dim);

/* ---- Early Termination ---- */
int hv_hamming_early(const_hv_t a, const_hv_t b, int best_so_far, int margin, int hv_words);
int hv_hamming_early_exit(const_hv_t a, const_hv_t b, int max_allowed, int hv_words);

/* ---- CompHD Dimension Folding ---- */
typedef uint64_t* hv_folded_t;
void hv_fold(hv_folded_t out, const_hv_t src, int fold, int hv_words);
int hv_hamming_folded(const uint64_t *a, const uint64_t *b, int hv_words_folded);

/* ---- Backend AVX2 (Simulated fallback or dynamic detection) ---- */
int hv_hamming_avx2(const_hv_t a, const_hv_t b, int hv_words);
void hv_bundle_avx2(hv_t out, const_hv_t *vecs, int n, int hv_words, int hv_dim);
int hdc_detect_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* HDC_CORE_H */