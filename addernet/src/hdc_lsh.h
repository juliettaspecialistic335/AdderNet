#ifndef HDC_LSH_H
#define HDC_LSH_H

#include "hdc_core.h"

/* default parameters for LSH */
#define LSH_K 16   /* number of hashes per table */
#define LSH_L 8    /* number of hash tables */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int k;          /* bits per bucket */
    int l;          /* number of tables */
    int n_classes;  
    int hv_words;
    int hv_dim;
    
    int **bit_indices; /* [L][K] - random bit indices to sample */
    
    uint64_t *codebook; /* copy of codebook [n_classes * hv_words] */
    
    int **hash_tables; /* [L][1<<K] -> index of class */
} hdc_lsh_t;

/* Build LSH index */
hdc_lsh_t* hdc_lsh_build_ex(const uint64_t *codebook, int n_classes, int k, int l, int hv_words, int hv_dim);

#define hdc_lsh_build(codebook, n_classes, hv_words, hv_dim) hdc_lsh_build_ex(codebook, n_classes, LSH_K, LSH_L, hv_words, hv_dim)

/* Query LSH: returns number of candidates written to 'candidates' array. */
int hdc_lsh_query(const hdc_lsh_t *lsh, const uint64_t* query, int *candidates, int max_candidates, int hv_words);

void hdc_lsh_free(hdc_lsh_t *lsh);

#ifdef __cplusplus
}
#endif

#endif /* HDC_LSH_H */