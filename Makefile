# AdderNet Library — Makefile
# ============================
#   make              Build both libraries (.so)
#   make addernet     Build libaddernet.so only
#   make hdc          Build libaddernet_hdc.so only
#   make test         Build and run all tests
#   make clean        Remove build artifacts

CC        = gcc
ARCH      := $(shell uname -m)
OS        := $(shell uname -s)

# OPT-7: Automatic platform detection
ifeq ($(ARCH), x86_64)
    SIMD_FLAGS = -mavx2 -mpopcnt -march=native
    SIMD_DEF   = -DHAVE_AVX2 -D__AVX2__
else ifeq ($(ARCH), aarch64)
    SIMD_FLAGS = -march=armv8-a+simd -mfpu=neon
    SIMD_DEF   = -DHAVE_NEON -D__ARM_NEON
else ifeq ($(ARCH), armv7l)
    SIMD_FLAGS = -mfpu=neon-vfpv4 -mfloat-abi=hard
    SIMD_DEF   = -DHAVE_NEON -D__ARM_NEON
else
    SIMD_FLAGS =
    SIMD_DEF   =
endif

CFLAGS    = -O3 -ffast-math -funroll-loops -fPIC -Wall -Wextra $(SIMD_FLAGS) $(SIMD_DEF)
LDFLAGS   = -lm -lpthread -fopenmp
SRC_DIR   = src
BUILD_DIR = build
TESTS_DIR = tests

# --- AdderNet (single-variable) ---
LIB_SRC  = $(SRC_DIR)/addernet.c
LIB_HDR  = $(SRC_DIR)/addernet.h
LIB_SO   = $(BUILD_DIR)/libaddernet.so
LIB_OBJ  = $(BUILD_DIR)/addernet.o

# --- AdderNet-HDC (multivariate) ---
HDC_CORE_SRC = $(SRC_DIR)/hdc_core.c
HDC_CORE_HDR = $(SRC_DIR)/hdc_core.h
HDC_LSH_SRC  = $(SRC_DIR)/hdc_lsh.c
HDC_LSH_HDR  = $(SRC_DIR)/hdc_lsh.h
HDC_SRC      = $(SRC_DIR)/addernet_hdc.c
HDC_HDR      = $(SRC_DIR)/addernet_hdc.h
HDC_SO       = $(BUILD_DIR)/libaddernet_hdc.so
HDC_CORE_OBJ = $(BUILD_DIR)/hdc_core.o
HDC_LSH_OBJ  = $(BUILD_DIR)/hdc_lsh.o
HDC_OBJ      = $(BUILD_DIR)/addernet_hdc.o

.PHONY: all addernet hdc cuda test test_addernet test_hdc clean

all: addernet hdc

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

addernet: $(LIB_SO)

$(LIB_OBJ): $(LIB_SRC) $(LIB_HDR) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(LIB_SRC) -o $(LIB_OBJ)

$(LIB_SO): $(LIB_OBJ)
	$(CC) -shared -o $(LIB_SO) $(LIB_OBJ) $(LDFLAGS)

hdc: $(HDC_SO)

$(HDC_CORE_OBJ): $(HDC_CORE_SRC) $(HDC_CORE_HDR) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(HDC_CORE_SRC) -o $(HDC_CORE_OBJ)

$(HDC_LSH_OBJ): $(HDC_LSH_SRC) $(HDC_LSH_HDR) $(HDC_CORE_HDR) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(HDC_LSH_SRC) -o $(HDC_LSH_OBJ)

$(HDC_OBJ): $(HDC_SRC) $(HDC_HDR) $(HDC_CORE_HDR) $(HDC_LSH_HDR) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(HDC_SRC) -o $(HDC_OBJ)

$(HDC_SO): $(HDC_CORE_OBJ) $(HDC_LSH_OBJ) $(HDC_OBJ)
	$(CC) -shared -o $(HDC_SO) $(HDC_CORE_OBJ) $(HDC_LSH_OBJ) $(HDC_OBJ) $(LDFLAGS)

# --- Tests ---

test: test_addernet test_hdc

test_addernet: $(LIB_SO) $(TESTS_DIR)/test_main.c
	$(CC) -O3 -march=native -I$(SRC_DIR) -o $(BUILD_DIR)/test_addernet \
		$(TESTS_DIR)/test_main.c -L$(BUILD_DIR) -laddernet $(LDFLAGS) -Wl,-rpath,$(BUILD_DIR)
	$(BUILD_DIR)/test_addernet

test_hdc: $(HDC_SO) $(TESTS_DIR)/test_hdc_main.c
	$(CC) -O3 -march=native -I$(SRC_DIR) -o $(BUILD_DIR)/test_hdc \
		$(TESTS_DIR)/test_hdc_main.c -L$(BUILD_DIR) -laddernet_hdc $(LDFLAGS) -Wl,-rpath,$(BUILD_DIR)
	$(BUILD_DIR)/test_hdc

# --- CUDA (inline PTX, no nvcc required) ---

CUDA_SRC = $(SRC_DIR)/hdc_core_cuda.c
CUDA_BATCH_SRC = $(SRC_DIR)/hdc_cuda_batch.c
CUDA_OBJ = $(BUILD_DIR)/hdc_core_cuda.o
CUDA_BATCH_OBJ = $(BUILD_DIR)/hdc_cuda_batch.o
CUDA_SO  = $(BUILD_DIR)/libaddernet_hdc_cuda.so

cuda: $(CUDA_SO)

$(CUDA_OBJ): $(CUDA_SRC) $(HDC_CORE_HDR) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(CUDA_SRC) -o $(CUDA_OBJ)

$(CUDA_BATCH_OBJ): $(CUDA_BATCH_SRC) $(HDC_CORE_HDR) $(HDC_HDR) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(CUDA_BATCH_SRC) -o $(CUDA_BATCH_OBJ)

$(CUDA_SO): $(HDC_CORE_OBJ) $(CUDA_OBJ) $(CUDA_BATCH_OBJ) $(HDC_OBJ)
	$(CC) -shared -o $(CUDA_SO) $(HDC_CORE_OBJ) $(CUDA_OBJ) $(CUDA_BATCH_OBJ) $(HDC_OBJ) $(LDFLAGS) -ldl

# --- CUDA (nvcc native build) ---
CUDA_NATIVE_SRC = $(SRC_DIR)/addernet_cuda.cu
CUDA_NATIVE_SO  = $(BUILD_DIR)/libaddernet_cuda.so

NVCC := $(shell command -v nvcc 2> /dev/null)

ifdef NVCC
all: addernet hdc cuda_native

$(CUDA_NATIVE_SO): $(CUDA_NATIVE_SRC) $(HDC_CORE_HDR) $(HDC_HDR) | $(BUILD_DIR)
	nvcc -O3 -Xcompiler -fPIC -shared $(CUDA_NATIVE_SRC) -o $(CUDA_NATIVE_SO) -I$(SRC_DIR) $(HDC_OBJ) $(HDC_CORE_OBJ) $(HDC_LSH_OBJ)

cuda_native: $(CUDA_NATIVE_SO)
else
cuda_native:
	@echo "nvcc not found. Skipping libaddernet_cuda.so build."
endif

clean:
	rm -rf $(BUILD_DIR)
