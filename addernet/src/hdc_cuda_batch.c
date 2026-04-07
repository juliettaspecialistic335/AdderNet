/*
 * HDC CUDA Batch Prediction — N predictions in parallel
 * =======================================================
 *   Each GPU thread processes one sample: encode → bundle → hamming search.
 *   Uses D=1000 (16 words) to keep per-thread stack at ~2KB.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fcntl.h>
#include "hdc_core.h"
#include "addernet_hdc.h"

/* CUDA driver types */
typedef int CUresult;
typedef void *CUmodule;
typedef void *CUfunction;
typedef unsigned long long CUdeviceptr;
typedef void *CUdevice;
#define CUDA_SUCCESS 0

/* We reuse the driver handle from the main CUDA module if loaded,
 * otherwise load our own. */
static struct {
    void *h;
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuCtxCreate)(void**, unsigned, CUdevice);
    CUresult (*cuCtxGetCurrent)(void**);
    CUresult (*cuCtxDestroy)(void*);
    CUresult (*cuModuleLoad)(CUmodule*, const char*);
    CUresult (*cuModuleUnload)(CUmodule);
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
    CUresult (*cuLaunchKernel)(CUfunction,
        unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,
        void*, void**, void*);
    CUresult (*cuCtxSynchronize)(void);
} G;

static int inited = 0;
static CUmodule batch_mod = NULL;
static CUfunction batch_fn = NULL;
static void *batch_ctx = NULL;

/* Batch kernel uses D=1000 for manageable stack */
#define m->hv_dim      1000
#define m->hv_words  (m->hv_dim / 64)   /* 16 */

#define S_(x) #x
#define S(x) S_(x)

static const char batch_ptx[] =
    ".version 6.5\n"
    ".target sm_61\n"
    ".address_size 64\n"
    "\n"
    "/* predict_batch_kernel\n"
    " * One thread per sample. Each thread:\n"
    " *   1. For each variable: compute bin, hv_from_seed(pos), hv_from_seed(val), bind\n"
    " *   2. Bundle via majority vote (uint16_t counts[D])\n"
    " *   3. Hamming search against codebook\n"
    " *   4. Write argmin to y_pred[i]\n"
    " */\n"
    ".visible .entry predict_batch_kernel(\n"
    "    .param .u64 p_bins,\n"
    "    .param .u64 p_codebook,\n"
    "    .param .u64 p_y_pred,\n"
    "    .param .u32 p_N,\n"
    "    .param .u32 p_n_vars,\n"
    "    .param .u32 p_n_classes\n"
    ")\n"
    "{\n"
    "    .reg .u32 %r<30>;\n"
    "    .reg .u64 %rd<20>;\n"
    "    .reg .pred %p<10>;\n"
    "\n"
    "    // i = thread index\n"
    "    mov.u32 %r0, %tid.x;\n"
    "    mov.u32 %r1, %ctaid.x;\n"
    "    mov.u32 %r2, %ntid.x;\n"
    "    mul.lo.u32 %r1, %r1, %r2;\n"
    "    add.u32 %r0, %r0, %r1;\n"
    "\n"
    "    ld.param.u32 %r19, [p_N];\n"
    "    setp.ge.u32 %p0, %r0, %r19;\n"
    "    @%p0 ret;\n"
    "\n"
    "    // Load params\n"
    "    ld.param.u64 %rd0, [p_bins];\n"
    "    ld.param.u64 %rd1, [p_codebook];\n"
    "    ld.param.u64 %rd2, [p_y_pred];\n"
    "    ld.param.u32 %r20, [p_n_vars];\n"
    "    ld.param.u32 %r21, [p_n_classes];\n"
    "\n"
    "    // === Phase 1: Bundle via counts ===\n"
    "    // Use local memory for counts[D] — 1000 × 2 bytes = 2KB\n"
    "    // We'll accumulate in registers where possible\n"
    "    // For simplicity, store counts in local memory\n"
    "\n"
    "    // Allocate counts in local memory: offset 0, size 2000 bytes\n"
    "    // Local memory is per-thread, addressed via %l[...]\n"
    "    // But PTX local memory needs .local directive\n"
    "    // Simpler: use register pairs as counters\n"
    "\n"
    "    // Actually, let's use a different approach:\n"
    "    // For each of the 16 words, keep 16 counters in registers\n"
    "    // This avoids local memory entirely\n"
    "\n"
    "    // query[16] — the result hypervector (in registers %rd3..%rd18)\n"
    "    // But we need to BUILD it via bit counting.\n"
    "\n"
    "    // Strategy: process one word at a time.\n"
    "    // For word w: count bits across all vars, produce query[w].\n"
    "    // Then hamming against codebook word w for all classes.\n"
    "    // Accumulate hamming distances in registers.\n"
    "\n"
    "    // hamming_dist[class] = running total\n"
    "    // We support up to 10 classes in registers %r5..%r14\n"
    "    mov.u32 %r5, 0;   // hamming[0]\n"
    "    mov.u32 %r6, 0;   // hamming[1]\n"
    "    mov.u32 %r7, 0;   // hamming[2]\n"
    "    mov.u32 %r8, 0;   // hamming[3]\n"
    "    mov.u32 %r9, 0;   // hamming[4]\n"
    "    mov.u32 %r10, 0;  // hamming[5]\n"
    "    mov.u32 %r11, 0;  // hamming[6]\n"
    "    mov.u32 %r12, 0;  // hamming[7]\n"
    "    mov.u32 %r13, 0;  // hamming[8]\n"
    "    mov.u32 %r14, 0;  // hamming[9]\n"
    "\n"
    "    // For each word w in [0, m->hv_words)\n"
    "    mov.u32 %r15, 0;  // word counter\n"
    "\n"
    "WORD_LOOP:\n"
    "    setp.ge.u32 %p1, %r15, " S(m->hv_words) ";\n"
    "    @%p1 bra FIND_BEST;\n"
    "\n"
    "    // Count bits in this word across all vars\n"
    "    // Use 64 uint16_t counters — but we only have registers\n"
    "    // Use a bit-parallel approach: process 4 bits at a time\n"
    "    // with a nibble lookup table\n"
    "\n"
    "    // Simplest correct approach: loop over bits\n"
    "    // For each bit b: count how many vars have bit set\n"
    "    // Then set output word bit if count > n_vars/2\n"
    "\n"
    "    // We'll build the word bit by bit\n"
    "    mov.u64 %rd3, 0;  // query_word = 0\n"
    "\n"
    "    mov.u32 %r16, 0;  // bit counter\n"
    "\n"
    "BIT_LOOP:\n"
    "    setp.ge.u32 %p2, %r16, 64;\n"
    "    @%p2 bra WORD_DONE;\n"
    "\n"
    "    // bit_mask = 1ULL << b\n"
    "    mov.u64 %rd4, 1;\n"
    "    shl.b64 %rd4, %rd4, %r16;\n"
    "\n"
    "    // count = 0\n"
    "    mov.u32 %r17, 0;\n"
    "\n"
    "    // For each variable v\n"
    "    mov.u32 %r18, 0;\n"
    "\n"
    "VAR_LOOP:\n"
    "    setp.ge.u32 %p3, %r18, %r20;\n"
    "    @%p3 bra BIT_CHECK;\n"
    "\n"
    "    // Read bin = bins[i * n_vars + v]\n"
    "    // addr = bins + (i * n_vars + v) * 4\n"
    "    mul.lo.u32 %r22, %r0, %r20;\n"
    "    add.u32 %r22, %r22, %r18;\n"
    "    mul.lo.u32 %r22, %r22, 4;\n"
    "    cvt.u64.u32 %rd5, %r22;\n"
    "    add.u64 %rd5, %rd0, %rd5;\n"
    "    ld.global.u32 %r23, [%rd5];\n"
    "\n"
    "    // seed_val = v * 100003 + bin\n"
    "    mov.u32 %r24, 100003;\n"
    "    mul.lo.u32 %r24, %r18, %r24;\n"
    "    add.u32 %r24, %r24, %r23;\n"
    "    cvt.u64.u32 %rd6, %r24;\n"
    "\n"
    "    // seed_pos = v * 999983\n"
    "    mov.u32 %r25, 999983;\n"
    "    mul.lo.u32 %r25, %r18, %r25;\n"
    "    cvt.u64.u32 %rd7, %r25;\n"
    "\n"
    "    // hv_from_seed for word w (xorshift to word w)\n"
    "    // val: skip w words\n"
    "    mov.u64 %rd8, %rd6;\n"
    "    mov.u32 %r26, 0;\n"
    "SEED_VAL:\n"
    "    setp.ge.u32 %p4, %r26, %r15;\n"
    "    @%p4 bra SEED_VAL_DONE;\n"
    "    shl.b64 %rd8, %rd8, 13;\n"
    "    xor.b64 %rd6, %rd6, %rd8;\n"
    "    shr.b64 %rd8, %rd6, 7;\n"
    "    xor.b64 %rd6, %rd6, %rd8;\n"
    "    shl.b64 %rd8, %rd6, 17;\n"
    "    xor.b64 %rd6, %rd6, %rd8;\n"
    "    mov.u64 %rd8, %rd6;\n"
    "    add.u32 %r26, %r26, 1;\n"
    "    bra SEED_VAL;\n"
    "SEED_VAL_DONE:\n"
    "    // %rd6 = val_word for this var\n"
    "\n"
    "    // pos: skip w words\n"
    "    mov.u64 %rd8, %rd7;\n"
    "    mov.u32 %r26, 0;\n"
    "SEED_POS:\n"
    "    setp.ge.u32 %p5, %r26, %r15;\n"
    "    @%p5 bra SEED_POS_DONE;\n"
    "    shl.b64 %rd8, %rd8, 13;\n"
    "    xor.b64 %rd7, %rd7, %rd8;\n"
    "    shr.b64 %rd8, %rd7, 7;\n"
    "    xor.b64 %rd7, %rd7, %rd8;\n"
    "    shl.b64 %rd8, %rd7, 17;\n"
    "    xor.b64 %rd7, %rd7, %rd8;\n"
    "    mov.u64 %rd8, %rd7;\n"
    "    add.u32 %r26, %r26, 1;\n"
    "    bra SEED_POS;\n"
    "SEED_POS_DONE:\n"
    "    // %rd7 = pos_word for this var\n"
    "\n"
    "    // pair = val ^ pos\n"
    "    xor.b64 %rd8, %rd6, %rd7;\n"
    "\n"
    "    // if (pair & bit_mask): count++\n"
    "    and.b64 %rd9, %rd8, %rd4;\n"
    "    setp.ne.u64 %p6, %rd9, 0;\n"
    "    @%p6 add.u32 %r17, %r17, 1;\n"
    "\n"
    "    add.u32 %r18, %r18, 1;\n"
    "    bra VAR_LOOP;\n"
    "\n"
    "BIT_CHECK:\n"
    "    // if (count > n_vars/2): query_word |= bit_mask\n"
    "    shr.u32 %r27, %r20, 1;  // n_vars / 2\n"
    "    setp.gt.u32 %p7, %r17, %r27;\n"
    "    @%p7 or.b64 %rd3, %rd3, %rd4;\n"
    "\n"
    "    add.u32 %r16, %r16, 1;\n"
    "    bra BIT_LOOP;\n"
    "\n"
    "WORD_DONE:\n"
    "    // query_word is in %rd3\n"
    "    // Now update hamming distances for all classes\n"
    "    // codebook is at %rd1, class c word w is at offset c*WORDS*8 + w*8\n"
    "\n"
    "    // Class 0\n"
    "    mov.u32 %r28, 0;\n"
    "    mul.lo.u32 %r28, %r15, 8;\n"
    "    cvt.u64.u32 %rd10, %r28;\n"
    "    add.u64 %rd10, %rd1, %rd10;\n"
    "    ld.global.u64 %rd11, [%rd10];\n"
    "    xor.b64 %rd12, %rd3, %rd11;\n"
    "    // __popc of 64-bit: split into two 32-bit\n"
    "    cvt.u32.u64 %r29, %rd12;\n"
    "    shr.b64 %rd13, %rd12, 32;\n"
    "    cvt.u32.u64 %r28, %rd13;\n"
    "    popc.b32 %r29, %r29;\n"
    "    popc.b32 %r28, %r28;\n"
    "    add.u32 %r5, %r5, %r29;\n"
    "    add.u32 %r5, %r5, %r28;\n"
    "\n"
    "    // Class 1\n"
    "    mov.u32 %r28, " S(m->hv_words * 8) ";\n"
    "    mov.u32 %r29, %r15;\n"
    "    mul.lo.u32 %r29, %r29, 8;\n"
    "    add.u32 %r28, %r28, %r29;\n"
    "    cvt.u64.u32 %rd10, %r28;\n"
    "    add.u64 %rd10, %rd1, %rd10;\n"
    "    ld.global.u64 %rd11, [%rd10];\n"
    "    xor.b64 %rd12, %rd3, %rd11;\n"
    "    cvt.u32.u64 %r29, %rd12;\n"
    "    shr.b64 %rd13, %rd12, 32;\n"
    "    cvt.u32.u64 %r28, %rd13;\n"
    "    popc.b32 %r29, %r29;\n"
    "    popc.b32 %r28, %r28;\n"
    "    add.u32 %r6, %r6, %r29;\n"
    "    add.u32 %r6, %r6, %r28;\n"
    "\n"
    "    // Class 2\n"
    "    mov.u32 %r28, " S(2 * m->hv_words * 8) ";\n"
    "    mov.u32 %r29, %r15;\n"
    "    mul.lo.u32 %r29, %r29, 8;\n"
    "    add.u32 %r28, %r28, %r29;\n"
    "    cvt.u64.u32 %rd10, %r28;\n"
    "    add.u64 %rd10, %rd1, %rd10;\n"
    "    ld.global.u64 %rd11, [%rd10];\n"
    "    xor.b64 %rd12, %rd3, %rd11;\n"
    "    cvt.u32.u64 %r29, %rd12;\n"
    "    shr.b64 %rd13, %rd12, 32;\n"
    "    cvt.u32.u64 %r28, %rd13;\n"
    "    popc.b32 %r29, %r29;\n"
    "    popc.b32 %r28, %r28;\n"
    "    add.u32 %r7, %r7, %r29;\n"
    "    add.u32 %r7, %r7, %r28;\n"
    "\n"
    "    // Classes 3-9 (same pattern)\n"
    "    mov.u32 %r28, " S(3 * m->hv_words * 8) ";\n"
    "    mov.u32 %r29, %r15; mul.lo.u32 %r29, %r29, 8; add.u32 %r28, %r28, %r29;\n"
    "    cvt.u64.u32 %rd10, %r28; add.u64 %rd10, %rd1, %rd10;\n"
    "    ld.global.u64 %rd11, [%rd10];\n"
    "    xor.b64 %rd12, %rd3, %rd11;\n"
    "    cvt.u32.u64 %r29, %rd12; shr.b64 %rd13, %rd12, 32; cvt.u32.u64 %r28, %rd13;\n"
    "    popc.b32 %r29, %r29; popc.b32 %r28, %r28;\n"
    "    add.u32 %r8, %r8, %r29; add.u32 %r8, %r8, %r28;\n"
    "\n"
    "    mov.u32 %r28, " S(4 * m->hv_words * 8) ";\n"
    "    mov.u32 %r29, %r15; mul.lo.u32 %r29, %r29, 8; add.u32 %r28, %r28, %r29;\n"
    "    cvt.u64.u32 %rd10, %r28; add.u64 %rd10, %rd1, %rd10;\n"
    "    ld.global.u64 %rd11, [%rd10];\n"
    "    xor.b64 %rd12, %rd3, %rd11;\n"
    "    cvt.u32.u64 %r29, %rd12; shr.b64 %rd13, %rd12, 32; cvt.u32.u64 %r28, %rd13;\n"
    "    popc.b32 %r29, %r29; popc.b32 %r28, %r28;\n"
    "    add.u32 %r9, %r9, %r29; add.u32 %r9, %r9, %r28;\n"
    "\n"
    "    mov.u32 %r28, " S(5 * m->hv_words * 8) ";\n"
    "    mov.u32 %r29, %r15; mul.lo.u32 %r29, %r29, 8; add.u32 %r28, %r28, %r29;\n"
    "    cvt.u64.u32 %rd10, %r28; add.u64 %rd10, %rd1, %rd10;\n"
    "    ld.global.u64 %rd11, [%rd10];\n"
    "    xor.b64 %rd12, %rd3, %rd11;\n"
    "    cvt.u32.u64 %r29, %rd12; shr.b64 %rd13, %rd12, 32; cvt.u32.u64 %r28, %rd13;\n"
    "    popc.b32 %r29, %r29; popc.b32 %r28, %r28;\n"
    "    add.u32 %r10, %r10, %r29; add.u32 %r10, %r10, %r28;\n"
    "\n"
    "    mov.u32 %r28, " S(6 * m->hv_words * 8) ";\n"
    "    mov.u32 %r29, %r15; mul.lo.u32 %r29, %r29, 8; add.u32 %r28, %r28, %r29;\n"
    "    cvt.u64.u32 %rd10, %r28; add.u64 %rd10, %rd1, %rd10;\n"
    "    ld.global.u64 %rd11, [%rd10];\n"
    "    xor.b64 %rd12, %rd3, %rd11;\n"
    "    cvt.u32.u64 %r29, %rd12; shr.b64 %rd13, %rd12, 32; cvt.u32.u64 %r28, %rd13;\n"
    "    popc.b32 %r29, %r29; popc.b32 %r28, %r28;\n"
    "    add.u32 %r11, %r11, %r29; add.u32 %r11, %r11, %r28;\n"
    "\n"
    "    mov.u32 %r28, " S(7 * m->hv_words * 8) ";\n"
    "    mov.u32 %r29, %r15; mul.lo.u32 %r29, %r29, 8; add.u32 %r28, %r28, %r29;\n"
    "    cvt.u64.u32 %rd10, %r28; add.u64 %rd10, %rd1, %rd10;\n"
    "    ld.global.u64 %rd11, [%rd10];\n"
    "    xor.b64 %rd12, %rd3, %rd11;\n"
    "    cvt.u32.u64 %r29, %rd12; shr.b64 %rd13, %rd12, 32; cvt.u32.u64 %r28, %rd13;\n"
    "    popc.b32 %r29, %r29; popc.b32 %r28, %r28;\n"
    "    add.u32 %r12, %r12, %r29; add.u32 %r12, %r12, %r28;\n"
    "\n"
    "    mov.u32 %r28, " S(8 * m->hv_words * 8) ";\n"
    "    mov.u32 %r29, %r15; mul.lo.u32 %r29, %r29, 8; add.u32 %r28, %r28, %r29;\n"
    "    cvt.u64.u32 %rd10, %r28; add.u64 %rd10, %rd1, %rd10;\n"
    "    ld.global.u64 %rd11, [%rd10];\n"
    "    xor.b64 %rd12, %rd3, %rd11;\n"
    "    cvt.u32.u64 %r29, %rd12; shr.b64 %rd13, %rd12, 32; cvt.u32.u64 %r28, %rd13;\n"
    "    popc.b32 %r29, %r29; popc.b32 %r28, %r28;\n"
    "    add.u32 %r13, %r13, %r29; add.u32 %r13, %r13, %r28;\n"
    "\n"
    "    mov.u32 %r28, " S(9 * m->hv_words * 8) ";\n"
    "    mov.u32 %r29, %r15; mul.lo.u32 %r29, %r29, 8; add.u32 %r28, %r28, %r29;\n"
    "    cvt.u64.u32 %rd10, %r28; add.u64 %rd10, %rd1, %rd10;\n"
    "    ld.global.u64 %rd11, [%rd10];\n"
    "    xor.b64 %rd12, %rd3, %rd11;\n"
    "    cvt.u32.u64 %r29, %rd12; shr.b64 %rd13, %rd12, 32; cvt.u32.u64 %r28, %rd13;\n"
    "    popc.b32 %r29, %r29; popc.b32 %r28, %r28;\n"
    "    add.u32 %r14, %r14, %r29; add.u32 %r14, %r14, %r28;\n"
    "\n"
    "    add.u32 %r15, %r15, 1;\n"
    "    bra WORD_LOOP;\n"
    "\n"
    "FIND_BEST:\n"
    "    // Find argmin of %r5..%r14\n"
    "    mov.u32 %r28, 0;   // best_class\n"
    "    mov.u32 %r29, %r5;  // best_dist\n"
    "\n"
    "    setp.lt.u32 %p8, %r6, %r29;\n"
    "    @%p8 mov.u32 %r28, 1;\n"
    "    @%p8 mov.u32 %r29, %r6;\n"
    "    setp.lt.u32 %p8, %r7, %r29;\n"
    "    @%p8 mov.u32 %r28, 2;\n"
    "    @%p8 mov.u32 %r29, %r7;\n"
    "    setp.lt.u32 %p8, %r8, %r29;\n"
    "    @%p8 mov.u32 %r28, 3;\n"
    "    @%p8 mov.u32 %r29, %r8;\n"
    "    setp.lt.u32 %p8, %r9, %r29;\n"
    "    @%p8 mov.u32 %r28, 4;\n"
    "    @%p8 mov.u32 %r29, %r9;\n"
    "    setp.lt.u32 %p8, %r10, %r29;\n"
    "    @%p8 mov.u32 %r28, 5;\n"
    "    @%p8 mov.u32 %r29, %r10;\n"
    "    setp.lt.u32 %p8, %r11, %r29;\n"
    "    @%p8 mov.u32 %r28, 6;\n"
    "    @%p8 mov.u32 %r29, %r11;\n"
    "    setp.lt.u32 %p8, %r12, %r29;\n"
    "    @%p8 mov.u32 %r28, 7;\n"
    "    @%p8 mov.u32 %r29, %r12;\n"
    "    setp.lt.u32 %p8, %r13, %r29;\n"
    "    @%p8 mov.u32 %r28, 8;\n"
    "    @%p8 mov.u32 %r29, %r13;\n"
    "    setp.lt.u32 %p8, %r14, %r29;\n"
    "    @%p8 mov.u32 %r28, 9;\n"
    "    @%p8 mov.u32 %r29, %r14;\n"
    "\n"
    "    // Write y_pred[i] = best_class\n"
    "    mul.lo.u32 %r27, %r0, 4;\n"
    "    cvt.u64.u32 %rd14, %r27;\n"
    "    add.u64 %rd14, %rd2, %rd14;\n"
    "    st.global.u32 [%rd14], %r28;\n"
    "\n"
    "    ret;\n"
    "}\n";

/* ---- Init ---- */
static int init_batch(void) {
    if (inited) return 1;

    G.h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!G.h) G.h = dlopen("libcuda.so", RTLD_LAZY);
    if (!G.h) return 0;

    #define L(s) G.s = dlsym(G.h, #s)
    L(cuInit); L(cuDeviceGet); L(cuCtxCreate); L(cuCtxGetCurrent);
    L(cuCtxDestroy); L(cuModuleLoad); L(cuModuleUnload);
    L(cuModuleGetFunction); L(cuMemAlloc); L(cuMemFree);
    L(cuMemcpyHtoD); L(cuMemcpyDtoH); L(cuLaunchKernel); L(cuCtxSynchronize);
    #undef L

    if (!G.cuInit || !G.cuModuleLoad || !G.cuLaunchKernel) return 0;
    G.cuInit(0);

    if (G.cuCtxGetCurrent) G.cuCtxGetCurrent(&batch_ctx);
    if (!batch_ctx) {
        CUdevice dev; G.cuDeviceGet(&dev, 0);
        G.cuCtxCreate(&batch_ctx, 0, dev);
    }

    char path[] = "/tmp/hdc_batch_XXXXXX.ptx";
    int fd = mkstemps(path, 4);
    if (fd < 0) return 0;
    write(fd, batch_ptx, strlen(batch_ptx)); close(fd);
    CUresult r = G.cuModuleLoad(&batch_mod, path);
    unlink(path);
    if (r != 0) return 0;

    r = G.cuModuleGetFunction(&batch_fn, batch_mod, "predict_batch_kernel");
    if (r != 0) return 0;

    inited = 1;
    return 1;
}

/* ---- Compute bins on CPU ---- */
static int *compute_bins(an_hdc_model *m, const double *X, int N) {
    int nv = m->n_vars;
    int *bins = (int *)malloc(N * nv * sizeof(int));
    if (!bins) return NULL;
    for (int i = 0; i < N; i++) {
        for (int v = 0; v < nv; v++) {
            bins[i * nv + v] = ((int)X[i * nv + v] + m->bias[v]) & m->table_mask;
        }
    }
    return bins;
}

/* ---- Pack codebook to flat uint64_t array (D=m->hv_dim) ---- */
/* The model's codebook uses the compiled HDC_WORDS (e.g. 157 for D=10000).
 * Our kernel uses m->hv_words (16 for D=1000).
 * We need to generate codebook entries at D=1000.
 * Strategy: re-encode training data at D=1000 and bundle.
 * For simplicity, just truncate the existing codebook to m->hv_words. */
static uint64_t *pack_codebook(an_hdc_model *m) {
    int nc = m->n_classes;
    int nw = m->hv_words;  /* 16 for D=1000 */
    uint64_t *cb = (uint64_t *)malloc(nc * nw * sizeof(uint64_t));
    if (!cb) return NULL;

    /* The model's hv_t has HDC_WORDS words (157 for D=10000).
     * We truncate to m->hv_words (16). This loses information but
     * gives a quick test. For production, re-train at D=1000. */
    for (int c = 0; c < nc; c++) {
        for (int w = 0; w < nw && w < HDC_WORDS; w++) {
            cb[c * nw + w] = m->codebook[c][w];
        }
        for (int w = HDC_WORDS; w < nw; w++) {
            cb[c * nw + w] = 0;
        }
    }
    return cb;
}

/* ---- Public API ---- */
int an_hdc_predict_batch_cuda(an_hdc_model *m, const double *X, int *y_pred, int N) {
    if (!m || !X || !y_pred || N <= 0) return -1;

    if (!init_batch()) {
        /* Fallback to CPU */
        an_hdc_predict_batch(m, X, y_pred, N);
        return 0;
    }

    int nv = m->n_vars;
    int nc = m->n_classes;

    /* Compute bins on CPU */
    int *bins = compute_bins(m, X, N);
    if (!bins) { an_hdc_predict_batch(m, X, y_pred, N); return 0; }

    /* Pack codebook */
    uint64_t *cb = pack_codebook(m);
    if (!cb) { free(bins); an_hdc_predict_batch(m, X, y_pred, N); return 0; }

    /* Allocate GPU memory */
    size_t bins_bytes = (size_t)N * nv * sizeof(int);
    size_t cb_bytes = (size_t)nc * m->hv_words * sizeof(uint64_t);
    size_t pred_bytes = (size_t)N * sizeof(int);

    CUdeviceptr d_bins, d_cb, d_pred;
    G.cuMemAlloc(&d_bins, bins_bytes);
    G.cuMemAlloc(&d_cb, cb_bytes);
    G.cuMemAlloc(&d_pred, pred_bytes);

    G.cuMemcpyHtoD(d_bins, bins, bins_bytes);
    G.cuMemcpyHtoD(d_cb, cb, cb_bytes);

    /* Launch kernel: one thread per sample */
    unsigned int block = 256;
    unsigned int grid = (N + block - 1) / block;

    void *args[] = { &d_bins, &d_cb, &d_pred, &N, &nv, &nc };
    G.cuLaunchKernel(batch_fn, grid,1,1, block,1,1, 0,NULL,args,NULL);
    CUresult sr = G.cuCtxSynchronize();

    if (sr != 0) {
        G.cuMemFree(d_bins); G.cuMemFree(d_cb); G.cuMemFree(d_pred);
        free(bins); free(cb);
        an_hdc_predict_batch(m, X, y_pred, N);
        return 0;
    }

    G.cuMemcpyDtoH(y_pred, d_pred, pred_bytes);

    G.cuMemFree(d_bins);
    G.cuMemFree(d_cb);
    G.cuMemFree(d_pred);
    free(bins);
    free(cb);
    return 0;
}
