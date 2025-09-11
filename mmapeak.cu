#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <mma.h>
#if (__CUDA_ARCH__ >= 890) && (CUDA_VERSION >= 12080)
#include <cuda_fp8.h>
#endif
#if (__CUDA_ARCH__ >= 1200) && (CUDA_VERSION >= 12080)
#include <cuda_fp4.h>
#endif

using namespace nvcuda::wmma;

#define N_LOOP_INTERNAL 8192
#define N_LOOP_CALIB 128
#define DEFAULT_TARGET_TIME 3.0f
#define cudaCheckError() cudaCheckError_(__FILE__, __LINE__)
inline void cudaCheckError_(const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

inline int _ConvertSMVer2Cores(int major, int minor) {
  typedef struct {
    int SM;
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {0x90, 128},
      {0xa0, 128},
      {0xa1, 128},
      {0xa3, 128},
      {0xb0, 128},
      {0xc0, 128},
      {0xc1, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }
  return nGpuArchCoresPerSM[index - 1].Cores;
}

template <typename InputType, typename OutputType, unsigned M, unsigned N, unsigned K>
__device__ void mma_(OutputType *data) {
    fragment<accumulator, M, N, K, OutputType> d;
    fragment<matrix_a, M, N, K, InputType, row_major> a;
    fragment<matrix_b, M, N, K, InputType, col_major> b;
    fill_fragment(d, 0);
    fill_fragment(a, 0);
    fill_fragment(b, 0);
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        mma_sync(d, a, b, d);
        __syncwarp();
    }
    OutputType *ptr = &data[threadIdx.y * M * N];
    store_matrix_sync(ptr, d, N, mem_row_major);
}

#if (__CUDA_ARCH__ >= 750)
inline __device__ void mma_s4_(
    fragment<accumulator, 8, 8, 32, int> &d,
    const fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major> &a,
    const fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major> &b,
    const fragment<accumulator, 8, 8, 32, int> &c) {
    asm volatile(
        "mma.sync.aligned.row.col.m8n8k32.s32.s4.s4.s32 "
        "{%0, %1}, {%2}, {%3}, {%4, %5};"
        : "=r"(d.x[0]), "=r"(d.x[1])
        : "r"(a.x[0]), "r"(b.x[0]), "r"(c.x[0]), "r"(c.x[1]));
}

template <typename InputType, typename OutputType, unsigned M, unsigned N, unsigned K>
__device__ void mma_s4_(OutputType *data) {
    fragment<accumulator, M, N, K, OutputType> d;
    fragment<matrix_a, M, N, K, InputType, row_major> a;
    fragment<matrix_b, M, N, K, InputType, col_major> b;
    fill_fragment(d, 0);
    fill_fragment(a, 0);
    fill_fragment(b, 0);
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        mma_s4_(d, a, b, d);
        __syncwarp();
    }
    OutputType *ptr = &data[threadIdx.y * M * N];
    store_matrix_sync(ptr, d, N, mem_row_major);
}
#endif

#if (__CUDA_ARCH__ >= 890) && (CUDA_VERSION >= 12080)
__device__ void mma_f8f8f16_16_8_32_(half *data) {
    uint32_t d[2] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
            "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
            : "=r"(d[0]), "=r"(d[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]));
        __syncwarp();
    }
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data), "r"(d[0]), "r"(d[1]));
}

__device__ void mma_f8f8f32_16_8_32_(float *data) {
    uint32_t d[4] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
        __syncwarp();
    }
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data), "r"(d[0]), "r"(d[1]));
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data + 64), "r"(d[2]), "r"(d[3]));
}
#endif

#if (__CUDA_ARCH__ >= 1200) && (CUDA_VERSION >= 12080)
__device__ void mma_mxf4mxf4f32_16_8_64_(float *data) {
    uint32_t d[4] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    uint32_t sa = 0;
    uint32_t sb = 0;
    static constexpr uint16_t bid_a = 0;
    static constexpr uint16_t tid_a = 0;
    static constexpr uint16_t bid_b = 0;
    static constexpr uint16_t tid_b = 0;
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, %14, {%15, %16}, %17, {%18, %19};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]),
              "r"(sa), "h"(bid_a), "h"(tid_a),
              "r"(sb), "h"(bid_b), "h"(tid_b));
        __syncwarp();
    }
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data), "r"(d[0]), "r"(d[1]));
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data + 64), "r"(d[2]), "r"(d[3]));
}

__device__ void mma_nvf4nvf4f32_16_8_64_(float *data) {
    uint32_t d[4] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    uint32_t sa = 0;
    uint32_t sb = 0;
    static constexpr uint16_t bid_a = 0;
    static constexpr uint16_t tid_a = 0;
    static constexpr uint16_t bid_b = 0;
    static constexpr uint16_t tid_b = 0;
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, %14, {%15, %16}, %17, {%18, %19};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]),
              "r"(sa), "h"(bid_a), "h"(tid_a),
              "r"(sb), "h"(bid_b), "h"(tid_b));
        __syncwarp();
    }
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data), "r"(d[0]), "r"(d[1]));
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data + 64), "r"(d[2]), "r"(d[3]));
}

__device__ void mma_f4f4f16_16_8_32_(half *data) {
    uint32_t d[2] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e2m1.e2m1.f16 "
			"{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
            : "=r"(d[0]), "=r"(d[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]));
        __syncwarp();
    }
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data), "r"(d[0]), "r"(d[1]));
}

__device__ void mma_f4f4f32_16_8_32_(float *data) {
    uint32_t d[4] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
			"{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
        __syncwarp();
    }
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data), "r"(d[0]), "r"(d[1]));
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data + 64), "r"(d[2]), "r"(d[3]));
}

__device__ void mma_f6f6f16_16_8_32_(half *data) {
    uint32_t d[2] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e3m2.e3m2.f16 "
			"{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"
            : "=r"(d[0]), "=r"(d[1])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]));
        __syncwarp();
    }
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data), "r"(d[0]), "r"(d[1]));
}

__device__ void mma_f6f6f32_16_8_32_(float *data) {
    uint32_t d[4] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e3m2.f32 "
			"{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
        __syncwarp();
    }
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data), "r"(d[0]), "r"(d[1]));
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data + 64), "r"(d[2]), "r"(d[3]));
}

__device__ void mma_mxf6mxf6f32_16_8_32_(float *data) {
    uint32_t d[4] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    uint32_t sa = 0;
    uint32_t sb = 0;
    static constexpr uint16_t bid_a = 0;
    static constexpr uint16_t tid_a = 0;
    static constexpr uint16_t bid_b = 0;
    static constexpr uint16_t tid_b = 0;
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e3m2.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, %14, {%15, %16}, %17, {%18, %19};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]),
              "r"(sa), "h"(bid_a), "h"(tid_a),
              "r"(sb), "h"(bid_b), "h"(tid_b));
        __syncwarp();
    }
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data), "r"(d[0]), "r"(d[1]));
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data + 64), "r"(d[2]), "r"(d[3]));
}

__device__ void mma_mxf8mxf8f32_16_8_32_(float *data) {
    uint32_t d[4] = {0};
    uint32_t a[4] = {0};
    uint32_t b[2] = {0};
    uint32_t sa = 0;
    uint32_t sb = 0;
    static constexpr uint16_t bid_a = 0;
    static constexpr uint16_t tid_a = 0;
    static constexpr uint16_t bid_b = 0;
    static constexpr uint16_t tid_b = 0;
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, %14, {%15, %16}, %17, {%18, %19};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]),
              "r"(sa), "h"(bid_a), "h"(tid_a),
              "r"(sb), "h"(bid_b), "h"(tid_b));
        __syncwarp();
    }
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data), "r"(d[0]), "r"(d[1]));
    asm volatile("wmma.store.d.sync.aligned.row.m8n8k32.global.s32 [%0], {%1, %2};" ::"l"(data + 64), "r"(d[2]), "r"(d[3]));
}
#endif

__global__ void mma_f16f16f16_16_16_16(void *data, int *rc) {
    mma_<half, half, 16, 16, 16>((half *)data);
    *rc = 0;
}

__global__ void mma_f16f16f16_32_8_16(void *data, int *rc) {
    mma_<half, half, 32, 8, 16>((half *)data);
    *rc = 0;
}

__global__ void mma_f16f16f32_16_16_16(void *data, int *rc) {
    mma_<half, float, 16, 16, 16>((float *)data);
    *rc = 0;
}

__global__ void mma_f16f16f32_32_8_16(void *data, int *rc) {
    mma_<half, float, 32, 8, 16>((float *)data);
    *rc = 0;
}

#if (__CUDA_ARCH__ >= 750)
__global__ void mma_s4s4s32_8_8_32(void *data, int *rc) {
    mma_s4_<experimental::precision::s4, int, 8, 8, 32>((int *)data);
    *rc = 0;
}

__global__ void mma_s8s8s32_16_16_16(void *data, int *rc) {
    mma_<signed char, int, 16, 16, 16>((int *)data);
    *rc = 0;
}

__global__ void mma_s8s8s32_32_8_16(void *data, int *rc) {
    mma_<signed char, int, 32, 8, 16>((int *)data);
    *rc = 0;
}
#else
__global__ void mma_s4s4s32_8_8_32(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_s8s8s32_16_16_16(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_s8s8s32_32_8_16(void *data, int *rc) {
    *rc = 1;
}
#endif

#if (__CUDA_ARCH__ >= 800) && (CUDA_VERSION >= 11010)
__global__ void mma_bf16bf16f32_16_16_16(void *data, int *rc) {
    mma_<__nv_bfloat16, float, 16, 16, 16>((float *)data);
    *rc = 0;
}

__global__ void mma_bf16bf16f32_32_8_16(void *data, int *rc) {
    mma_<__nv_bfloat16, float, 32, 8, 16>((float *)data);
    *rc = 0;
}

__global__ void mma_tf32tf32f32_16_16_8(void *data, int *rc) {
    mma_<precision::tf32, float, 16, 16, 8>((float *)data);
    *rc = 0;
}
#else
__global__ void mma_bf16bf16f32_16_16_16(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_bf16bf16f32_32_8_16(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_tf32tf32f32_16_16_8(void *data, int *rc) {
    *rc = 1;
}
#endif

#if (__CUDA_ARCH__ >= 890) && (CUDA_VERSION >= 12080)
__global__ void mma_f8f8f16_16_8_32(void *data, int *rc) {
    mma_f8f8f16_16_8_32_((half *)data);
    *rc = 0;
}

__global__ void mma_f8f8f32_16_8_32(void *data, int *rc) {
    mma_f8f8f32_16_8_32_((float *)data);
    *rc = 0;
}
#else
__global__ void mma_f8f8f16_16_8_32(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_f8f8f32_16_8_32(void *data, int *rc) {
    *rc = 1;
}
#endif

#if (__CUDA_ARCH__ >= 1200) && (CUDA_VERSION >= 12080)
__global__ void mma_mxf4mxf4f32_16_8_64(void *data, int *rc) {
    mma_mxf4mxf4f32_16_8_64_((float *)data);
    *rc = 0;
}

__global__ void mma_nvf4nvf4f32_16_8_64(void *data, int *rc) {
    mma_nvf4nvf4f32_16_8_64_((float *)data);
    *rc = 0;
}

__global__ void mma_f4f4f16_16_8_32(void *data, int *rc) {
    mma_f4f4f16_16_8_32_((half *)data);
    *rc = 0;
}

__global__ void mma_f4f4f32_16_8_32(void *data, int *rc) {
    mma_f4f4f32_16_8_32_((float *)data);
    *rc = 0;
}

__global__ void mma_f6f6f16_16_8_32(void *data, int *rc) {
    mma_f6f6f16_16_8_32_((half *)data);
    *rc = 0;
}

__global__ void mma_f6f6f32_16_8_32(void *data, int *rc) {
    mma_f6f6f32_16_8_32_((float *)data);
    *rc = 0;
}

__global__ void mma_mxf6mxf6f32_16_8_32(void *data, int *rc) {
    mma_mxf6mxf6f32_16_8_32_((float *)data);
    *rc = 0;
}

__global__ void mma_mxf8mxf8f32_16_8_32(void *data, int *rc) {
    mma_mxf8mxf8f32_16_8_32_((float *)data);
    *rc = 0;
}
#else
__global__ void mma_mxf4mxf4f32_16_8_64(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_nvf4nvf4f32_16_8_64(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_f4f4f16_16_8_32(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_f4f4f32_16_8_32(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_f6f6f16_16_8_32(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_f6f6f32_16_8_32(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_mxf6mxf6f32_16_8_32(void *data, int *rc) {
    *rc = 1;
}

__global__ void mma_mxf8mxf8f32_16_8_32(void *data, int *rc) {
    *rc = 1;
}
#endif

template <typename OutputType, unsigned M, unsigned N, unsigned K>
void run(void *kernel, float targetTime) {
    const int num_tb = 512;
    const int num_warps_per_tb = 4;
    const int warp_size = 32;
    dim3 grid(num_tb);
    dim3 block(warp_size, num_warps_per_tb);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaCheckError();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaCheckError();
    void *data = nullptr;
    size_t nbytes = num_warps_per_tb * M * N * sizeof(OutputType);
    cudaMalloc(&data, nbytes);
    cudaCheckError();

    int *d_rc;
    cudaMalloc(&d_rc, sizeof(int));
    ((void (*)(void *, int *))kernel)<<<grid, block, 0, stream>>>(data, d_rc);
    int h_rc = 0;
    cudaMemcpy(&h_rc, d_rc, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_rc != 0) {
        printf("  not supported\n");
    }
    else {
        int n_loop = N_LOOP_CALIB;
        cudaCheckError();
        cudaEventRecord(start, stream);
        for (int i = 0; i < n_loop; i++) {
            ((void (*)(void *, int *))kernel)<<<grid, block, 0, stream>>>(data, d_rc);
        }
        cudaCheckError();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        n_loop = (int)(targetTime * 1000 / ms * n_loop);
        n_loop = n_loop > 0 ? n_loop : N_LOOP_CALIB;

        cudaEventRecord(start, stream);
        for (int i = 0; i < n_loop; i++) {
            ((void (*)(void *, int *))kernel)<<<grid, block, 0, stream>>>(data, d_rc);
        }
        cudaCheckError();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaStreamDestroy(stream);
        cudaCheckError();
        cudaEventElapsedTime(&ms, start, stop);
        float ops = 1.0f * num_tb * num_warps_per_tb * n_loop * N_LOOP_INTERNAL * M * N * K * 2;
        printf("%.1f Tflops\n", ops / ms / 1.0e9f);
    }
    cudaFree(d_rc);
    cudaFree(data);
    cudaCheckError();
}

void print_usage() {
    printf("Usage: mmapeak [options]\n");
    printf("Options:\n");
    printf("  -t <seconds>   Set target time for benchmarks in seconds (default: %.1f)\n", DEFAULT_TARGET_TIME);
    printf("  -h, --help     Show this help message\n");
}

int main(int argc, char **argv) {
    float targetTime = DEFAULT_TARGET_TIME;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            targetTime = atof(argv[++i]);
            if (targetTime <= 0) {
                printf("Error: Target time must be positive\n");
                print_usage();
                return 1;
            }
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        }
        else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaCheckError();
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 1;
    }
    for (int i = 0; i < deviceCount; i++) {
        printf("----------------------------------------\n");

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        cudaCheckError();
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  CUDA Cores: %d\n", _ConvertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount);
        int clockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, i);
        printf("  GPU Max Clock rate: %.0f MHz\n", clockRate * 1e-3f);
        printf("  Total global memory: %.1f GiB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

        cudaSetDevice(i);
        cudaCheckError();

        printf("Running benchmarks with target time: %.1f seconds\n", targetTime);


        if (prop.major * 10 + prop.minor >= 75) {
			printf("mma_s4s4s32_8_8_32:       ");
			run<int, 8, 8, 32>((void *)mma_s4s4s32_8_8_32, targetTime);
			if ((prop.major * 10 + prop.minor >= 89) && (CUDA_VERSION >= 12080)) {
				if (prop.major >= 12) {
					printf("mma_mxf4mxf4f32_16_8_64:  ");
					run<float, 16, 8, 64>((void *)mma_mxf4mxf4f32_16_8_64, targetTime);
					printf("mma_nvf4nvf4f32_16_8_64:  ");
					run<float, 16, 8, 64>((void *)mma_nvf4nvf4f32_16_8_64, targetTime);
					printf("mma_f4f4f16_16_8_32:      ");
					run<half, 16, 8, 32>((void *)mma_f4f4f16_16_8_32, targetTime);
					printf("mma_f4f4f32_16_8_32:      ");
					run<float, 16, 8, 32>((void *)mma_f4f4f32_16_8_32, targetTime);
					printf("mma_f6f6f16_16_8_32:      ");
					run<half, 16, 8, 32>((void *)mma_f6f6f16_16_8_32, targetTime);
					printf("mma_f6f6f32_16_8_32:      ");
					run<float, 16, 8, 32>((void *)mma_f6f6f32_16_8_32, targetTime);
					printf("mma_mxf6mxf6f32_16_8_32:  ");
					run<float, 16, 8, 32>((void *)mma_mxf6mxf6f32_16_8_32, targetTime);
					printf("mma_mxf8mxf8f32_16_8_32:  ");
					run<float, 16, 8, 32>((void *)mma_mxf8mxf8f32_16_8_32, targetTime);
				}
				printf("mma_f8f8f16_16_8_32:      ");
				run<half, 16, 8, 32>((void *)mma_f8f8f16_16_8_32, targetTime);
				printf("mma_f8f8f32_16_8_32:      ");
				run<float, 16, 8, 32>((void *)mma_f8f8f32_16_8_32, targetTime);
			}
			printf("mma_s8s8s32_16_16_16:     ");
			run<int, 16, 16, 16>((void *)mma_s8s8s32_16_16_16, targetTime);
			printf("mma_s8s8s32_32_8_16:      ");
			run<int, 32, 8, 16>((void *)mma_s8s8s32_32_8_16, targetTime);
        }
		if (prop.major >= 8) {
			printf("mma_bf16bf16f32_16_16_16: ");
			run<float, 16, 16, 16>((void *)mma_bf16bf16f32_16_16_16, targetTime);
			printf("mma_bf16bf16f32_32_8_16:  ");
			run<float, 32, 8, 16>((void *)mma_bf16bf16f32_32_8_16, targetTime);
		}
        printf("mma_f16f16f16_16_16_16:   ");
        run<half, 16, 16, 16>((void *)mma_f16f16f16_16_16_16, targetTime);
        printf("mma_f16f16f16_32_8_16:    ");
        run<half, 32, 8, 16>((void *)mma_f16f16f16_32_8_16, targetTime);
        printf("mma_f16f16f32_16_16_16:   ");
        run<float, 16, 16, 16>((void *)mma_f16f16f32_16_16_16, targetTime);
        printf("mma_f16f16f32_32_8_16:    ");
        run<float, 32, 8, 16>((void *)mma_f16f16f32_32_8_16, targetTime);
		if (prop.major >= 8) {
			printf("mma_tf32tf32f32_16_16_8:  ");
			run<float, 16, 16, 8>((void *)mma_tf32tf32f32_16_16_8, targetTime);
		}
    }
    return 0;
}