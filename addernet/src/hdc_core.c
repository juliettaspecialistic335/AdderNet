#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include "hdc_core.h"

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

/* ---- Internal RNG (xorshift64) ---- */

static uint64_t rng_state = 0xDEADBEEFCAFEBABEULL;

void hv_seed(unsigned int seed) {
    rng_state = (uint64_t)seed * 6364136223846793005ULL + 1442695040888963407ULL;
    if (rng_state == 0) rng_state = 1;
}

uint64_t hv_seed_state(void) { return rng_state; }
void hv_seed_set(uint64_t state) { rng_state = state ? state : 1; }

static inline uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

/* ---- Core Operations ---- */

void hv_from_seed(hv_t out, uint64_t seed, int hv_words) {
    uint64_t old_state = rng_state;
    hv_seed_set(seed);
    for (int i = 0; i < hv_words; i++) {
        out[i] = xorshift64();
    }
    rng_state = old_state;
}

void hv_from_seed_fast(hv_t out, uint64_t seed, int hv_words, int hv_dim) {
    uint64_t s = seed;
    for (int i = 0; i < hv_words; i++) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out[i] = s;
    }
    uint64_t mask = (hv_dim % 64 == 0) ? ~0ULL : (1ULL << (hv_dim % 64)) - 1;
    out[hv_words - 1] &= mask;
}

void hv_random(hv_t out, int hv_words, int hv_dim) {
    for (int i = 0; i < hv_words; i++) {
        out[i] = xorshift64();
    }
    uint64_t mask = (hv_dim % 64 == 0) ? ~0ULL : (1ULL << (hv_dim % 64)) - 1;
    out[hv_words - 1] &= mask;
}

void hv_bind(hv_t out, const_hv_t a, const_hv_t b, int hv_words) {
    for (int i = 0; i < hv_words; i++) {
        out[i] = a[i] ^ b[i];
    }
}

int hv_hamming(const_hv_t a, const_hv_t b, int hv_words) {
    int dist = 0;
    for (int i = 0; i < hv_words; i++) {
        dist += __builtin_popcountll(a[i] ^ b[i]);
    }
    return dist;
}

int hv_hamming_unrolled(const_hv_t a, const_hv_t b, int hv_words) {
    int dist = 0;
    int w = 0;
    for (; w <= hv_words - 4; w += 4) {
        dist += __builtin_popcountll(a[w+0] ^ b[w+0]);
        dist += __builtin_popcountll(a[w+1] ^ b[w+1]);
        dist += __builtin_popcountll(a[w+2] ^ b[w+2]);
        dist += __builtin_popcountll(a[w+3] ^ b[w+3]);
    }
    for (; w < hv_words; w++) dist += __builtin_popcountll(a[w] ^ b[w]);
    return dist;
}

void hv_hamming_batch4(const uint64_t *queries, const uint64_t *ref, int *dists, int hv_words) {
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int w = 0; w < hv_words; w++) {
        uint64_t r = ref[w];
        d0 += __builtin_popcountll(queries[0 * hv_words + w] ^ r);
        d1 += __builtin_popcountll(queries[1 * hv_words + w] ^ r);
        d2 += __builtin_popcountll(queries[2 * hv_words + w] ^ r);
        d3 += __builtin_popcountll(queries[3 * hv_words + w] ^ r);
    }
    dists[0] += d0; dists[1] += d1; dists[2] += d2; dists[3] += d3;
}

float hv_similarity(const_hv_t a, const_hv_t b, int hv_words, int hv_dim) {
    return 1.0f - (float)hv_hamming(a, b, hv_words) / (float)hv_dim;
}

void hv_copy(hv_t dst, const_hv_t src, int hv_words) {
    for (int i = 0; i < hv_words; i++) dst[i] = src[i];
}

void hv_zero(hv_t dst, int hv_words) {
    for (int i = 0; i < hv_words; i++) dst[i] = 0;
}

void hv_add_noise(hv_t out, const_hv_t src, float temperature, int hv_words) {
    for (int i = 0; i < hv_words; i++) out[i] = src[i];
    if (temperature <= 0.0f) return;
    
    int threshold = (int)(temperature * 256.0f);
    for (int w = 0; w < hv_words; w++) {
        for (int b = 0; b < 64; b++) {
            if ((rand() & 255) < threshold) {
                out[w] ^= (1ULL << b);
            }
        }
    }
}

void hv_from_hadamard(hv_t out, int var, int bin, int hv_words, int hv_dim) {
    uint64_t seed = (uint64_t)var * 100003ULL + (uint64_t)bin;
    hv_from_seed(out, seed, hv_words);
    uint64_t mask = (hv_dim % 64 == 0) ? ~0ULL : (1ULL << (hv_dim % 64)) - 1;
    out[hv_words - 1] &= mask;
}

void hv_bundle(hv_t out, const_hv_t *vecs, int n, int hv_words, int hv_dim) {
    if (n == 0) { hv_zero(out, hv_words); return; }
    if (n == 1) { hv_copy(out, vecs[0], hv_words); return; }

    uint16_t *counts = (uint16_t *)malloc(hv_dim * sizeof(uint16_t));
    memset(counts, 0, hv_dim * sizeof(uint16_t));
    int threshold = n / 2;

    for (int i = 0; i < n; i++) {
        const_hv_t v = vecs[i];
        for (int w = 0; w < hv_words; w++) {
            uint64_t word = v[w];
            int base = w * 64;
            for (int bit = 0; bit < 64 && (base + bit) < hv_dim; bit++) {
                if ((word >> bit) & 1) counts[base + bit]++;
            }
        }
    }

    hv_zero(out, hv_words);
    for (int i = 0; i < hv_dim; i++) {
        if (counts[i] > threshold) {
            out[i / 64] |= (1ULL << (i % 64));
        } else if (counts[i] == threshold) {
            if (xorshift64() & 1) out[i / 64] |= (1ULL << (i % 64));
        }
    }
    free(counts);
}

void hv_bundle_weighted(hv_t out, const_hv_t *vecs, const int *weights, int n, int hv_words, int hv_dim) {
    if (n == 0) { hv_zero(out, hv_words); return; }

    int *counts = (int *)malloc(hv_dim * sizeof(int));
    memset(counts, 0, hv_dim * sizeof(int));
    int total_weight = 0;

    for (int i = 0; i < n; i++) {
        total_weight += weights[i];
        const_hv_t v = vecs[i];
        for (int w = 0; w < hv_words; w++) {
            uint64_t word = v[w];
            int base = w * 64;
            for (int bit = 0; bit < 64 && (base + bit) < hv_dim; bit++) {
                if ((word >> bit) & 1) counts[base + bit] += weights[i];
            }
        }
    }

    hv_zero(out, hv_words);
    int threshold = total_weight / 2;
    for (int i = 0; i < hv_dim; i++) {
        if (counts[i] > threshold) {
            out[i / 64] |= (1ULL << (i % 64));
        }
    }
    free(counts);
}

void hdc_circulant_init(hdc_circulant_t *circ, uint64_t seed_val, uint64_t seed_pos, int hv_words, int hv_dim) {
    (void)seed_pos;
    if (!circ->base_vector) circ->base_vector = (uint64_t *)malloc(hv_words * sizeof(uint64_t));
    hv_from_seed(circ->base_vector, seed_val, hv_words);
    uint64_t mask = (hv_dim % 64 == 0) ? ~0ULL : (1ULL << (hv_dim % 64)) - 1;
    circ->base_vector[hv_words - 1] &= mask;
    circ->circulant_ready = 1;
}

void hv_rotate(hv_t out, const_hv_t src, int shift, int hv_words, int hv_dim) {
    if (shift == 0) { hv_copy(out, src, hv_words); return; }
    shift = ((shift % hv_dim) + hv_dim) % hv_dim;
    int word_shift = shift / 64;
    int bit_shift = shift % 64;
    
    for (int i = 0; i < hv_words; i++) {
        int j = (i + word_shift) % hv_words;
        int k = (i + word_shift + 1) % hv_words;
        uint64_t w1 = src[j];
        uint64_t w2 = src[k];
        if (bit_shift == 0) {
            out[i] = w1;
        } else {
            out[i] = (w1 >> bit_shift) | (w2 << (64 - bit_shift));
        }
    }
    uint64_t mask = (hv_dim % 64 == 0) ? ~0ULL : (1ULL << (hv_dim % 64)) - 1;
    out[hv_words - 1] &= mask;
}

void hv_from_circulant(hv_t out, const hdc_circulant_t *circ, int var, int bin, int hv_words, int hv_dim) {
    int shift = (var * 13 + bin * 7) % hv_dim;
    hv_rotate(out, circ->base_vector, shift, hv_words, hv_dim);
}

int hv_hamming_early(const_hv_t a, const_hv_t b, int best_so_far, int margin, int hv_words) {
    int dist = 0;
    int max_allowed = best_so_far + margin;
    int w = 0;
    for (; w <= hv_words - 4; w += 4) {
        dist += __builtin_popcountll(a[w+0] ^ b[w+0]);
        dist += __builtin_popcountll(a[w+1] ^ b[w+1]);
        dist += __builtin_popcountll(a[w+2] ^ b[w+2]);
        dist += __builtin_popcountll(a[w+3] ^ b[w+3]);
        if (dist > max_allowed) return dist;
    }
    for (; w < hv_words; w++) {
        dist += __builtin_popcountll(a[w] ^ b[w]);
    }
    return dist;
}

int hv_hamming_early_exit(const_hv_t a, const_hv_t b, int max_allowed, int hv_words) {
    int dist = 0;
    for (int w = 0; w < hv_words; w++) {
        dist += __builtin_popcountll(a[w] ^ b[w]);
        if (dist > max_allowed) return dist;
    }
    return dist;
}

void hv_fold(hv_folded_t out, const_hv_t src, int fold, int hv_words) {
    int words_per_seg = hv_words / fold;
    int hv_words_folded = words_per_seg;
    hv_zero(out, hv_words_folded);
    for (int f = 0; f < fold; f++) {
        for (int w = 0; w < words_per_seg && w < hv_words_folded; w++) {
            out[w] ^= src[f * words_per_seg + w];
        }
    }
}

int hv_hamming_folded(const uint64_t *a, const uint64_t *b, int hv_words_folded) {
    int dist = 0;
    for (int w = 0; w < hv_words_folded; w++) dist += __builtin_popcountll(a[w] ^ b[w]);
    return dist;
}

int hv_hamming_avx2(const_hv_t a, const_hv_t b, int hv_words) { return hv_hamming_unrolled(a, b, hv_words); }
void hv_bundle_avx2(hv_t out, const_hv_t *vecs, int n, int hv_words, int hv_dim) { hv_bundle(out, vecs, n, hv_words, hv_dim); }

int hdc_detect_backend(void) { return 0; }

void hv_bundle_flat(hv_t out, const uint64_t *flat_vecs, int n, int hv_words, int hv_dim) {
    if (n == 0) { hv_zero(out, hv_words); return; }
    if (n == 1) { hv_copy(out, flat_vecs, hv_words); return; }
    uint16_t *counts = (uint16_t *)malloc(hv_dim * sizeof(uint16_t));
    memset(counts, 0, hv_dim * sizeof(uint16_t));
    int threshold = n / 2;
    for (int i = 0; i < n; i++) {
        const uint64_t *v = &flat_vecs[i * hv_words];
        for (int w = 0; w < hv_words; w++) {
            uint64_t word = v[w];
            int base = w * 64;
            for (int bit = 0; bit < 64 && (base + bit) < hv_dim; bit++) {
                if ((word >> bit) & 1) counts[base + bit]++;
            }
        }
    }
    hv_zero(out, hv_words);
    for (int i = 0; i < hv_dim; i++) {
        if (counts[i] > threshold) {
            out[i / 64] |= (1ULL << (i % 64));
        } else if (counts[i] == threshold) {
            if (xorshift64() & 1) out[i / 64] |= (1ULL << (i % 64));
        }
    }
    free(counts);
}
