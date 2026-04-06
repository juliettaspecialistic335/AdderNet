#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "hdc_core.h"
#include "addernet_hdc.h"

// ---------------------------------------------------------
// CUDA Kernel: Hamming Distance (L1) Search
// ---------------------------------------------------------
// Computes distance between query hypervectors and codebook.
// Uses native __popcll for fast binary L1 distance.
__global__ void hdc_predict_batch_kernel(const uint64_t* __restrict__ queries,
                                         const uint64_t* __restrict__ codebook,
                                         int* __restrict__ outputs,
                                         int N, int n_classes, int hdc_words) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int best_class = 0;
    int min_dist = 0x7FFFFFFF;

    const uint64_t* q = queries + i * hdc_words;

    for (int c = 0; c < n_classes; c++) {
        const uint64_t* cb = codebook + c * hdc_words;
        int dist = 0;
        for (int w = 0; w < hdc_words; w++) {
            dist += __popcll(q[w] ^ cb[w]);
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_class = c;
        }
    }
    outputs[i] = best_class;
}

// ---------------------------------------------------------
// C Wrapper for Python Integration
// ---------------------------------------------------------
// CPU performs encoding into hypervectors, GPU does parallel Hamming search
extern "C" int an_hdc_predict_batch_cuda(an_hdc_model *m, const double *X, int *y_pred, int N) {
    if (!m || !X || !y_pred || N <= 0) return -1;

    int hdc_words = m->hv_words;
    int hdc_dim = m->hv_dim;
    int n_classes = m->n_classes;

    // 1. Allocate host memory for queries
    uint64_t* h_queries = (uint64_t*)malloc(N * hdc_words * sizeof(uint64_t));
    if (!h_queries) return -1;

    // 2. Encode on CPU (Standard Value-Position Binding & Majority Vote)
    for (int i = 0; i < N; i++) {
        uint16_t *counts = (uint16_t *)calloc(hdc_dim, sizeof(uint16_t));
        const double* x = &X[i * m->n_vars];

        uint64_t *val_hv = (uint64_t *)malloc(hdc_words * sizeof(uint64_t));
        uint64_t *pair_hv = (uint64_t *)malloc(hdc_words * sizeof(uint64_t));
        for (int v = 0; v < m->n_vars; v++) {
            int bin = ((int)x[v] + m->bias[v]) & m->table_mask;
            hv_from_seed(val_hv, (uint64_t)v * 100003ULL + (uint64_t)bin, hdc_words);
            hv_bind(pair_hv, &m->position_hvs[v * hdc_words], val_hv, hdc_words);

            for (int w = 0; w < hdc_words; w++) {
                uint64_t word = pair_hv[w];
                for (int bit = 0; bit < 64; bit++) {
                    if ((w * 64 + bit) < hdc_dim && ((word >> bit) & 1)) {
                        counts[w * 64 + bit]++;
                    }
                }
            }
        }

        int threshold = m->n_vars / 2;
        for (int w = 0; w < hdc_words; w++) {
            uint64_t word = 0;
            for (int bit = 0; bit < 64; bit++) {
                if (w * 64 + bit < hdc_dim && counts[w * 64 + bit] > threshold) {
                    word |= (1ULL << bit);
                }
            }
            h_queries[i * hdc_words + w] = word;
        }
        free(counts);
        free(val_hv);
        free(pair_hv);
    }

    // 3. Allocate device memory
    uint64_t *d_queries, *d_codebook;
    int *d_outputs;
    size_t queries_size = N * hdc_words * sizeof(uint64_t);
    size_t codebook_size = n_classes * hdc_words * sizeof(uint64_t);
    size_t outputs_size = N * sizeof(int);

    cudaError_t err1 = cudaMalloc((void**)&d_queries, queries_size);
    cudaError_t err2 = cudaMalloc((void**)&d_codebook, codebook_size);
    cudaError_t err3 = cudaMalloc((void**)&d_outputs, outputs_size);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        free(h_queries);
        if (err1 == cudaSuccess) cudaFree(d_queries);
        if (err2 == cudaSuccess) cudaFree(d_codebook);
        if (err3 == cudaSuccess) cudaFree(d_outputs);
        return -1;
    }

    // 4. Copy data to device
    cudaMemcpy(d_queries, h_queries, queries_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_codebook, m->codebook, codebook_size, cudaMemcpyHostToDevice);

    // 5. Run the Kernel
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    hdc_predict_batch_kernel<<<grid_size, block_size>>>(d_queries, d_codebook, d_outputs, N, n_classes, hdc_words);

    cudaDeviceSynchronize();

    // 6. Copy result back
    cudaMemcpy(y_pred, d_outputs, outputs_size, cudaMemcpyDeviceToHost);

    // 7. Free device and host memory
    cudaFree(d_queries);
    cudaFree(d_codebook);
    cudaFree(d_outputs);
    free(h_queries);

    return 0;
}
