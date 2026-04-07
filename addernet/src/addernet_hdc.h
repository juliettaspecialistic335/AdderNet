/*
 * AdderNet-HDC — Multivariate classification using Hyperdimensional Computing
 */

#ifndef ADDERNET_HDC_H
#define ADDERNET_HDC_H

#include "hdc_core.h"
#include "hdc_lsh.h"

typedef struct {
    int i;
    int j;
} an_interaction_pair_t;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int        n_vars;
    int        n_classes;
    int        table_size;         /* encoding resolution (power of 2) */
    int        table_mask;         /* table_size - 1 */
    int        hv_dim;             /* <---- NEW: dynamic dimensionality */
    int        hv_words;           /* <---- NEW: (hv_dim + 63) / 64 */
    int       *bias;               /* bias[v] per variable */
    uint64_t  *position_hvs;       /* [n_vars * hv_words] role vectors */
    uint64_t  *codebook;           /* [n_classes * hv_words] class prototypes */
    char     **class_names;        /* [n_classes] or NULL */
    void      *cache;              /* OPT-1: encoding cache */
    int        use_cache;          /* 1 = use cache, 0 = compute on-the-fly */
    int        n_threads;          /* OPT-5: threads for batch prediction */
    
    hdc_circulant_t circulant;
    int               use_circulant;
    int               use_hadamard;
    
    int  use_early_term;
    int  early_term_margin;
    
    hv_folded_t *codebook_folded;
    int          use_folded;
    int          fold_factor;
    
    hdc_lsh_t *lsh_index;
    int         use_lsh;

    an_interaction_pair_t *interaction_pairs;
    int n_interaction_pairs;
} an_hdc_model;

an_hdc_model *an_hdc_create(int n_vars, int n_classes, int table_size, const int *bias, int hv_dim);
void an_hdc_free(an_hdc_model *m);

void an_hdc_train(an_hdc_model *m, const double *X, const int *y, int n_samples);
int an_hdc_retrain(an_hdc_model *m, const double *X, const int *y, int n_samples, int n_iter, float lr, int margin, float regen_rate, int patience, int verbose, int *epochs_run);

int an_hdc_predict(const an_hdc_model *m, const double *x);
int an_hdc_predict_batch(const an_hdc_model *m, const double *X, int *outputs, int n);
int an_hdc_predict_batch_avx(const an_hdc_model *m, const double *X, int *outputs, int n);
int an_hdc_predict_batch_mt(const an_hdc_model *m, const double *X, int *outputs, int n, int n_threads);

void an_hdc_warm_cache(an_hdc_model *m);
void an_hdc_set_cache(an_hdc_model *m, int use_cache);
void an_hdc_set_threads(an_hdc_model *m, int n_threads);

int an_hdc_save(const an_hdc_model *m, const char *path);
an_hdc_model *an_hdc_load(const char *path);

/* CUDA prototypes */
int an_hdc_predict_batch_cuda(an_hdc_model *m, const double *X, int *y_pred, int N);
int an_hdc_retrain_cuda(an_hdc_model *m, const double *X, const int *y, int n_samples, int n_iter, float lr, int margin, int patience, int verbose, int *epochs_run);

void an_hdc_set_circulant(an_hdc_model *m, int enable);
void an_hdc_set_hadamard(an_hdc_model *m, int enable);
void an_hdc_set_early_termination(an_hdc_model *m, int enable, int margin);

void an_hdc_fold_codebook(an_hdc_model *m, int fold);
int an_hdc_predict_folded(const an_hdc_model *m, const double *x);

void an_hdc_build_lsh(an_hdc_model *m);
void an_hdc_build_lsh_ex(an_hdc_model *m, int k, int l);
void an_hdc_set_lsh(an_hdc_model *m, int enable);

void an_hdc_predict_top_k(const an_hdc_model *m, const double *x, int *out_classes, int k);
void an_hdc_set_interactions(an_hdc_model *m, const int *pairs_i, const int *pairs_j, int n_pairs);

#ifdef __cplusplus
}
#endif

#endif /* ADDERNET_HDC_H */