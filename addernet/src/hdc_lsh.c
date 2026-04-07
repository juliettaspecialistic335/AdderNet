#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "hdc_lsh.h"

hdc_lsh_t* hdc_lsh_build_ex(const uint64_t *codebook, int n_classes, int k, int l, int hv_words, int hv_dim) {
    hdc_lsh_t *lsh = (hdc_lsh_t *)malloc(sizeof(hdc_lsh_t));
    lsh->k = k;
    lsh->l = l;
    lsh->n_classes = n_classes;
    lsh->hv_words = hv_words;
    lsh->hv_dim = hv_dim;
    
    lsh->codebook = (uint64_t *)malloc(n_classes * hv_words * sizeof(uint64_t));
    for (int i = 0; i < n_classes * hv_words; i++) lsh->codebook[i] = codebook[i];

    lsh->bit_indices = (int **)malloc(l * sizeof(int *));
    for (int tbl = 0; tbl < l; tbl++) {
        lsh->bit_indices[tbl] = (int *)malloc(k * sizeof(int));
        for (int bit = 0; bit < k; bit++) {
            lsh->bit_indices[tbl][bit] = rand() % hv_dim;
        }
    }

    int n_buckets = 1 << k;
    lsh->hash_tables = (int **)malloc(l * sizeof(int *));
    for (int tbl = 0; tbl < l; tbl++) {
        lsh->hash_tables[tbl] = (int *)malloc(n_buckets * sizeof(int));
        for (int i = 0; i < n_buckets; i++) lsh->hash_tables[tbl][i] = -1;

        for (int c = 0; c < n_classes; c++) {
            int hash_val = 0;
            for (int bit = 0; bit < k; bit++) {
                int bit_idx = lsh->bit_indices[tbl][bit];
                int word_idx = bit_idx / 64;
                int bit_pos = bit_idx % 64;
                int bit_val = (lsh->codebook[c * hv_words + word_idx] >> bit_pos) & 1U;
                if (bit_val) {
                    hash_val |= (1 << bit);
                }
            }
            if (lsh->hash_tables[tbl][hash_val] == -1) {
                lsh->hash_tables[tbl][hash_val] = c;
            } else {
                int curr_c = lsh->hash_tables[tbl][hash_val];
                int dist_curr = hv_hamming(&lsh->codebook[curr_c * hv_words], &lsh->codebook[c * hv_words], hv_words);
                if (dist_curr == 0) {
                    lsh->hash_tables[tbl][hash_val] = c;
                }
            }
        }
    }
    return lsh;
}

int hdc_lsh_query(const hdc_lsh_t *lsh, const uint64_t* query, int *candidates, int max_candidates, int hv_words) {
    (void)hv_words;  /* unused — hash derived from codebook, not raw HV */
    int n_cand = 0;
    int *visited = (int *)calloc(lsh->n_classes, sizeof(int));

    for (int tbl = 0; tbl < lsh->l; tbl++) {
        int hash_val = 0;
        for (int bit = 0; bit < lsh->k; bit++) {
            int bit_idx = lsh->bit_indices[tbl][bit];
            int word_idx = bit_idx / 64;
            int bit_pos = bit_idx % 64;
            int bit_val = (query[word_idx] >> bit_pos) & 1U;
            if (bit_val) {
                hash_val |= (1 << bit);
            }
        }
        int c = lsh->hash_tables[tbl][hash_val];
        if (c != -1 && !visited[c]) {
            visited[c] = 1;
            candidates[n_cand++] = c;
            if (n_cand >= max_candidates) break;
        }
    }
    free(visited);
    return n_cand;
}

void hdc_lsh_free(hdc_lsh_t *lsh) {
    if (!lsh) return;
    free(lsh->codebook);
    for (int tbl = 0; tbl < lsh->l; tbl++) {
        free(lsh->bit_indices[tbl]);
        free(lsh->hash_tables[tbl]);
    }
    free(lsh->bit_indices);
    free(lsh->hash_tables);
    free(lsh);
}
