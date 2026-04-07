/*
 * AdderNet-HDC CUDA Retraining Kernel
 * =====================================
 *   Implements AdaptHD/RefineHD iterative error correction on GPU.
 *
 *   Flow per epoch:
 *   1. Predict: GPU Hamming search of pre-encoded samples vs codebook
 *   2. Update: GPU updates cb_counts and cb_binary for mispredicted samples
 *
 *   Encoding is done on CPU (matching an_hdc_predict) before calling this
 *   function, since bundling with majority vote is complex to parallelize.
 *   The GPU handles the iterative search + update loop which is the actual
 *   bottleneck (many Hamming comparisons + bit-level updates per epoch).
 *
 *   Signature matches an_hdc_retrain_cuda from addernet_hdc.h:
 *     int an_hdc_retrain_cuda(an_hdc_model *m, const double *X, const int *y,
 *                             int n_samples, int n_iter, float lr,
 *                             int margin, int patience, int verbose,
 *                             int *epochs_run);
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "hdc_core.h"
#include "addernet_hdc.h"

#define MAX_WORDS 64

/* ==========================================================
 * Kernel: predict_hamming_search
 *   One thread per sample. Hamming search against codebook.
 *   Uses pre-encoded sample HVs (uploaded from CPU).
 * ========================================================== */
__global__ void predict_hamming_kernel(
    const uint64_t* __restrict__ sample_hvs,  /* [N * hv_words] pre-encoded */
    const uint64_t* __restrict__ codebook,    /* [n_classes * hv_words] */
    int* __restrict__ y_pred,                 /* [N] output */
    int N,
    int n_classes,
    int hv_words
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int best_class = 0;
    int min_dist = 0x7FFFFFFF;

    for (int c = 0; c < n_classes; c++) {
        int dist = 0;
        for (int w = 0; w < hv_words; w++) {
            dist += __popcll(sample_hvs[i * hv_words + w] ^ codebook[c * hv_words + w]);
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_class = c;
        }
    }

    y_pred[i] = best_class;
}

/* ==========================================================
 * Kernel: update_cb_counts
 *   One thread per mispredicted training sample.
 *   Uses sample_hvs for bit-level margin checking and count updates.
 * ========================================================== */
__global__ void update_cb_counts_kernel(
    const uint64_t* __restrict__ sample_hvs,  /* [N * hv_words] */
    const int*      __restrict__ y_pred,      /* [N] predictions */
    const int*      __restrict__ y_true,      /* [N] ground truth */
    int16_t*        __restrict__ cb_counts,   /* [n_classes * hv_dim] */
    uint64_t*       __restrict__ cb_binary,   /* [n_classes * hv_words] */
    int N,
    int n_classes,
    int hv_words,
    int hv_dim,
    int16_t lr_int,
    int margin,
    int* __restrict__ n_updates_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int pred = y_pred[i];
    int true_c = y_true[i];
    if (true_c < 0 || true_c >= n_classes) return;
    if (pred < 0 || pred >= n_classes) return;

    int do_update = 0;
    int penalize_c = pred;

    if (margin <= 0) {
        /* Pure AdaptHD: update only on misprediction */
        do_update = (pred != true_c);
    } else {
        /* RefineHD: update if margin is threatened */
        int dist_true = 0;
        int dist_comp = hv_dim + 1;
        int closest_comp = -1;

        for (int w = 0; w < hv_words; w++) {
            dist_true += __popcll(sample_hvs[i * hv_words + w] ^ cb_binary[(size_t)true_c * hv_words + w]);
        }
        for (int c = 0; c < n_classes; c++) {
            if (c == true_c) continue;
            int d = 0;
            for (int w = 0; w < hv_words; w++) {
                d += __popcll(sample_hvs[i * hv_words + w] ^ cb_binary[(size_t)c * hv_words + w]);
            }
            if (d < dist_comp) { dist_comp = d; closest_comp = c; }
        }
        if (dist_true + margin > dist_comp) {
            do_update = 1;
            if (pred != true_c) {
                penalize_c = pred;
            } else {
                penalize_c = (closest_comp >= 0) ? closest_comp : pred;
            }
        }
    }

    if (!do_update) return;

    atomicAdd(n_updates_out, 1);

    /* Update counts and binary for all set bits in sample_hv */
    for (int w = 0; w < hv_words; w++) {
        uint64_t word = sample_hvs[i * hv_words + w];
        int base = w * 64;
        /* Process bits 2 at a time for fewer iterations */
        for (int bit = 0; bit < 64; bit++) {
            if (word & (1ULL << bit)) {
                int idx = base + bit;
                /* CUDA 12.8 compatibility: atomicAdd requires int* for 32-bit */
                int *tc_ptr = (int *)&cb_counts[(size_t)true_c * hv_dim + idx];
                int *pc_ptr = (int *)&cb_counts[(size_t)penalize_c * hv_dim + idx];
                atomicAdd(tc_ptr, (int)lr_int);
                atomicAdd(pc_ptr, -(int)lr_int);

                /* In-kernel binary update: set if count > 0, clear otherwise */
                int tc_val = (int)cb_counts[(size_t)true_c * hv_dim + idx];
                int pc_val = (int)cb_counts[(size_t)penalize_c * hv_dim + idx];
                uint64_t mask = (1ULL << bit);

                /* CUDA 12.8 compatibility: atomicOr/And requires unsigned long long* */
                if (tc_val > 0)
                    atomicOr((unsigned long long *)&cb_binary[(size_t)true_c * hv_words + w], (unsigned long long)mask);
                else
                    atomicAnd((unsigned long long *)&cb_binary[(size_t)true_c * hv_words + w], (unsigned long long)~mask);

                if (pc_val > 0)
                    atomicOr((unsigned long long *)&cb_binary[(size_t)penalize_c * hv_words + w], (unsigned long long)mask);
                else
                    atomicAnd((unsigned long long *)&cb_binary[(size_t)penalize_c * hv_words + w], (unsigned long long)~mask);
            }
        }
    }
}

/* ==========================================================
 * Host: an_hdc_retrain_cuda
 * ========================================================== */
extern "C" int an_hdc_retrain_cuda(an_hdc_model *m, const double *X, const int *y,
                                   int n_samples, int n_iter, float lr,
                                   int margin, int patience, int verbose,
                                   int *epochs_run_out) {
    if (!m || !X || !y || n_samples <= 0 || n_iter <= 0) {
        if (epochs_run_out) *epochs_run_out = 0;
        return 0;
    }
    if (lr <= 0.0f) lr = 1.0f;
    if (margin < 0) margin = 0;

    /* Validation split (same as CPU: 75/25) */
    int n_val   = n_samples / 4;
    int n_train = n_samples - n_val;
    if (n_val < 1 && n_samples >= 2) { n_val = 1; n_train = n_samples - 1; }

    int nw = m->hv_words;
    int nd = m->hv_dim;
    int nc = m->n_classes;
    int nv = m->n_vars;

    /* ---- Pre-encode samples on CPU (matching an_hdc_predict logic) ---- */
    uint64_t *h_sample_hvs = (uint64_t *)aligned_alloc(64, n_samples * nw * sizeof(uint64_t));
    if (!h_sample_hvs) { if (epochs_run_out) *epochs_run_out = 0; return 0; }

    for (int i = 0; i < n_samples; i++) {
        const double *x = &X[i * nv];
        uint16_t *counts = (uint16_t *)calloc(nd, sizeof(uint16_t));
        if (!counts) { h_sample_hvs[i * nw + 0] = 0; continue; }

        for (int v = 0; v < nv; v++) {
            int bin = ((int)x[v] + m->bias[v]) & m->table_mask;

            /* Generate value HV words via xorshift */
            uint64_t seed = (uint64_t)v * 100003ULL + (uint64_t)bin;
            for (int w = 0; w < nw; w++) {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                uint64_t val_word = seed;
                uint64_t pair_word = m->position_hvs[v * nw + w] ^ val_word;

                uint64_t word = pair_word;
                int base = w * 64;
                while (word) {
                    int bit = __builtin_ctzll(word);
                    counts[base + bit]++;
                    word &= word - 1;
                }
            }
        }

        int threshold = nv / 2;
        for (int w = 0; w < nw; w++) {
            uint64_t word_out = 0;
            for (int bit = 0; bit < 64; bit++) {
                int idx = w * 64 + bit;
                if (idx < nd && counts[idx] > threshold)
                    word_out |= (1ULL << bit);
            }
            h_sample_hvs[i * nw + w] = word_out;
        }
        free(counts);
    }

    /* ---- Upload to GPU ---- */
    uint64_t *d_sample_hvs, *d_cb_binary;
    int16_t  *d_cb_counts;
    int      *d_y_pred, *d_y_true, *d_n_updates;

    size_t samples_sz = (size_t)n_samples * nw * sizeof(uint64_t);
    size_t cb_counts_sz = (size_t)nc * nd * sizeof(int16_t);
    size_t cb_binary_sz = nc * nw * sizeof(uint64_t);

    cudaMalloc((void **)&d_sample_hvs, samples_sz);
    cudaMemcpy(d_sample_hvs, h_sample_hvs, samples_sz, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_y_pred, n_train * sizeof(int));
    cudaMalloc((void **)&d_y_true, n_train * sizeof(int));
    cudaMemcpy(d_y_true, y, n_train * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_cb_counts, cb_counts_sz);
    cudaMalloc((void **)&d_cb_binary, cb_binary_sz);
    cudaMalloc((void **)&d_n_updates, sizeof(int));

    /* Initialize cb_counts from current codebook (+/- 1000) */
    {
        int16_t *h_cb_counts = (int16_t *)calloc(nc * nd, sizeof(int16_t));
        for (int c = 0; c < nc; c++) {
            for (int bit = 0; bit < nd; bit++) {
                int w = bit / 64, b = bit % 64;
                h_cb_counts[(size_t)c * nd + bit] =
                    (m->codebook[c * nw + w] & (1ULL << b)) ? 1000 : -1000;
            }
        }
        cudaMemcpy(d_cb_counts, h_cb_counts, cb_counts_sz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cb_binary, m->codebook, cb_binary_sz, cudaMemcpyHostToDevice);
        free(h_cb_counts);
    }

    /* ---- Early stopping state (CPU) ---- */
    double best_val_acc = 0.0;
    int best_epoch = 0, no_improve = 0, epochs_run = 0;

    int block = 256;

    /* ---- Iterative loop ---- */
    for (int it = 0; it < n_iter; it++) {
        /* Step 1: Predict (GPU Hamming search) */
        int grid = (n_train + block - 1) / block;
        predict_hamming_kernel<<<grid, block>>>(
            d_sample_hvs, d_cb_binary, d_y_pred, n_train, nc, nw
        );
        cudaDeviceSynchronize();

        /* Step 2: Update counts for mispredicted samples (GPU) */
        int16_t lr_int = (int16_t)(lr * 2.0f);
        cudaMemset(d_n_updates, 0, sizeof(int));
        update_cb_counts_kernel<<<grid, block>>>(
            d_sample_hvs, d_y_pred, d_y_true,
            d_cb_counts, d_cb_binary,
            n_train, nc, nw, nd, lr_int, margin, d_n_updates
        );

        int h_updates = 0;
        cudaMemcpy(&h_updates, d_n_updates, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if (h_updates == 0) {
            epochs_run = it + 1;
            break;
        }
        epochs_run = it + 1;

        /* Step 3: Validation accuracy (CPU — codebook is on GPU, sync it) */
        uint64_t *h_cb_tmp = (uint64_t *)malloc(nc * nw * sizeof(uint64_t));
        cudaMemcpy(h_cb_tmp, d_cb_binary, cb_binary_sz, cudaMemcpyDeviceToHost);

        uint64_t *saved_codebook = m->codebook;
        m->codebook = h_cb_tmp;

        int n_val_correct = 0;
        for (int i = n_train; i < n_samples; i++) {
            if (an_hdc_predict(m, &X[i * nv]) == y[i]) n_val_correct++;
        }
        double val_acc = (n_val > 0) ? (double)n_val_correct / (double)n_val : 0.0;
        m->codebook = saved_codebook;
        free(h_cb_tmp);

        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
            best_epoch = epochs_run;
            no_improve = 0;
        } else {
            no_improve++;
        }

        if (verbose > 0 && (epochs_run % verbose == 0 || epochs_run == 1)) {
            fprintf(stderr, "Epoch %4d/%d | val=%.1f%% | best=%.1f%% @%d\n",
                    epochs_run, n_iter,
                    val_acc * 100.0, best_val_acc * 100.0, best_epoch);
        }
        if (patience > 0 && no_improve >= patience) {
            if (verbose > 0)
                fprintf(stderr, "Early stop epoch %d — best val: %.1f%%\n",
                        best_epoch, best_val_acc * 100.0);
            break;
        }
    }

    /* Copy final codebook back */
    cudaMemcpy(m->codebook, d_cb_binary, cb_binary_sz, cudaMemcpyDeviceToHost);

    /* Cleanup */
    cudaFree(d_sample_hvs);
    cudaFree(d_y_pred);
    cudaFree(d_y_true);
    cudaFree(d_cb_counts);
    cudaFree(d_cb_binary);
    cudaFree(d_n_updates);
    free(h_sample_hvs);

    if (epochs_run_out) *epochs_run_out = epochs_run;
    return epochs_run;
}
