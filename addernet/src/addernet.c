/*
 * AdderNet Library — Implementation
 * ===================================
 *   Neural network using pure addition (zero multiplication at inference).
 *
 *   Compile: gcc -O3 -march=native -fPIC -shared -o libaddernet.so addernet.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "addernet.h"

/* ---- Internal helpers ---- */

typedef struct { int c; double f; } an_sample;

static int an_expand(const an_sample *src, int ns, an_sample *out, int lo, int hi) {
    an_sample s[256];
    if (ns > 256) ns = 256;
    memcpy(s, src, ns * sizeof(an_sample));

    /* Sort by c */
    for (int i = 0; i < ns - 1; i++)
        for (int j = i + 1; j < ns; j++)
            if (s[j].c < s[i].c) {
                an_sample t = s[i]; s[i] = s[j]; s[j] = t;
            }

    /* Slopes for extrapolation beyond edges */
    double sl = (s[1].f - s[0].f) / (double)(s[1].c - s[0].c);
    double sr = (s[ns-1].f - s[ns-2].f) / (double)(s[ns-1].c - s[ns-2].c);

    int n = 0;
    for (int v = lo; v <= hi; v++) {
        if (v <= s[0].c) {
            out[n].c = v;
            out[n].f = s[0].f + sl * (v - s[0].c);
            n++;
        } else if (v >= s[ns-1].c) {
            out[n].c = v;
            out[n].f = s[ns-1].f + sr * (v - s[ns-1].c);
            n++;
        } else {
            for (int i = 0; i < ns - 1; i++) {
                if (s[i].c <= v && v <= s[i+1].c) {
                    double d = s[i+1].c - s[i].c;
                    double frac = d > 0 ? (v - s[i].c) / d : 0;
                    out[n].c = v;
                    out[n].f = s[i].f + frac * (s[i+1].f - s[i].f);
                    n++;
                    break;
                }
            }
        }
    }
    return n;
}

static void an_train_samples(an_layer *layer, const an_sample *data, int n, int epochs) {
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < n; i++) {
            int idx = (data[i].c + layer->bias) & AN_TABLE_MASK;
            double err  = layer->offset[idx] - data[i].f;

            /* Try +lr */
            layer->offset[idx] += layer->lr;
            double eu = layer->offset[idx] - data[i].f;

            /* Try -lr */
            layer->offset[idx] -= 2.0 * layer->lr;
            double ed = layer->offset[idx] - data[i].f;

            /* Restore */
            layer->offset[idx] += layer->lr;

            /* Apply best direction */
            if (fabs(eu) < fabs(err))
                layer->offset[idx] += layer->lr;
            else if (fabs(ed) < fabs(err))
                layer->offset[idx] -= layer->lr;
        }
    }
}

/* ---- Public API ---- */

an_layer *an_layer_create(int size, int bias, int input_min, int input_max, double lr) {
    if (size <= 0 || (size & (size - 1)) != 0)
        return NULL;  /* size must be power of 2 */
    if (size > AN_TABLE_SIZE)
        return NULL;

    an_layer *layer = (an_layer *)an_aligned_alloc(64, sizeof(an_layer));
    if (!layer) return NULL;

    layer->size      = size;
    layer->bias      = bias;
    layer->input_min = input_min;
    layer->input_max = input_max;
    layer->lr        = lr;
    memset(layer->offset, 0, sizeof(double) * size);

    return layer;
}

void an_layer_free(an_layer *layer) {
    free(layer);
}

int an_train(an_layer *layer,
             const double *inputs, const double *targets,
             int n_samples, int epochs_raw, int epochs_expanded)
{
    if (!layer || !inputs || !targets || n_samples <= 0)
        return -1;

    /* Convert to internal sample format */
    an_sample *raw = (an_sample *)malloc(n_samples * sizeof(an_sample));
    if (!raw) return -1;
    for (int i = 0; i < n_samples; i++) {
        raw[i].c = (int)inputs[i];
        raw[i].f = targets[i];
    }

    /* Phase 1: train on raw samples */
    an_train_samples(layer, raw, n_samples, epochs_raw);

    /* Phase 2: expand and train on dense data */
    int expand_lo = layer->input_min;
    int expand_hi = layer->input_max;
    int expand_n  = expand_hi - expand_lo + 1;
    if (expand_n > 0 && epochs_expanded > 0) {
        an_sample *dense = (an_sample *)malloc(expand_n * sizeof(an_sample));
        if (dense) {
            int nd = an_expand(raw, n_samples, dense, expand_lo, expand_hi);
            an_train_samples(layer, dense, nd, epochs_expanded);
            free(dense);
        }
    }

    free(raw);
    return 0;
}

double an_predict(const an_layer *layer, double input) {
    int idx = ((int)input + layer->bias) & AN_TABLE_MASK;
    return layer->offset[idx];
}

int an_predict_batch(const an_layer *layer,
                     const double *inputs, double *outputs, int n)
{
    if (!layer || !inputs || !outputs || n <= 0)
        return -1;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++) {
        int idx = ((int)inputs[i] + layer->bias) & AN_TABLE_MASK;
        outputs[i] = layer->offset[idx];
    }
    return 0;
}

int an_save(const an_layer *layer, const char *path) {
    if (!layer || !path) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Write header */
    fwrite(&layer->size,      sizeof(int),    1, f);
    fwrite(&layer->bias,      sizeof(int),    1, f);
    fwrite(&layer->input_min, sizeof(int),    1, f);
    fwrite(&layer->input_max, sizeof(int),    1, f);
    fwrite(&layer->lr,        sizeof(double), 1, f);

    /* Write offset table */
    fwrite(layer->offset, sizeof(double), layer->size, f);

    fclose(f);
    return 0;
}

an_layer *an_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    int size, bias, input_min, input_max;
    double lr;

    if (fread(&size,      sizeof(int),    1, f) != 1) goto fail;
    if (fread(&bias,      sizeof(int),    1, f) != 1) goto fail;
    if (fread(&input_min, sizeof(int),    1, f) != 1) goto fail;
    if (fread(&input_max, sizeof(int),    1, f) != 1) goto fail;
    if (fread(&lr,        sizeof(double), 1, f) != 1) goto fail;

    an_layer *layer = an_layer_create(size, bias, input_min, input_max, lr);
    if (!layer) goto fail;

    if (fread(layer->offset, sizeof(double), size, f) != (size_t)size) {
        an_layer_free(layer);
        goto fail;
    }

    fclose(f);
    return layer;

fail:
    fclose(f);
    return NULL;
}

int an_get_offset(const an_layer *layer, double *buf, int buf_size) {
    if (!layer || !buf || buf_size < layer->size)
        return -1;
    memcpy(buf, layer->offset, layer->size * sizeof(double));
    return 0;
}

int an_get_size(const an_layer *layer) { return layer ? layer->size : 0; }
int an_get_bias(const an_layer *layer) { return layer ? layer->bias : 0; }
int an_get_input_min(const an_layer *layer) { return layer ? layer->input_min : 0; }
int an_get_input_max(const an_layer *layer) { return layer ? layer->input_max : 0; }
double an_get_lr(const an_layer *layer) { return layer ? layer->lr : 0.0; }

/* ---- P3B: mmap for LUT loading ---- */
#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

void *an_layer_mmap_load(const char *path) {
    if (!path) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }

    void *mapped = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    if (mapped == MAP_FAILED) return NULL;
    return mapped;
}
#endif
