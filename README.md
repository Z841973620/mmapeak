# MMAPEAK - CUDA Matrix Multiply Performance Benchmark

MMAPEAK is a CUDA-based benchmarking tool designed to measure the peak performance of matrix multiplication operations across various data types and tensor core configurations on NVIDIA GPUs.

## Overview

This tool measures the throughput of NVIDIA's Tensor Core dense operations using different precision formats:
- 4-bit integer (Int4)
- 4-bit floating point (FP4)
- 4-bit floating point with group scale (MXFP4 G32, NVFP4 G16)
- 6-bit floating point (FP6)
- 6-bit floating point with group scale (MXFP6 G32)
- 8-bit integer (INT8)
- 8-bit floating point (FP8)
- 8-bit floating point with group scale (MXFP8 G32)
- 16-bit floating point (FP16, BF16)
- 32-bit floating point (TF32)

## Building

### Using CMake

```bash
cmake -B build && cmake --build build -j
```

#### Note

Please use CUDA Toolkit version 12.8.1 (or later) instead of 12.8.0 to ensure compatibility with FP8/FP4 .

`wgmma` is not currently utilized, results in suboptimal FP8 performance on Hopper devices.

## Usage

```bash
./mmapeak [options]
```

### Options

- `-t <seconds>`: Set target time for benchmarks in seconds (default: 3.0)
- `-h, --help`: Show help message

## Example Output

```
----------------------------------------
Device 0: NVIDIA Thor
  Compute capability: 11.0
  Multiprocessors: 20
  CUDA Cores: 2560
  GPU Max Clock rate: 1049 MHz
  Total global memory: 122.8 GiB

Running benchmarks with target time: 3.0 seconds
mma_s4s4s32_8_8_32:       7.7 Tflops
mma_f8f8f16_16_8_32:      232.1 Tflops
mma_f8f8f32_16_8_32:      190.2 Tflops
mma_s8s8s32_16_16_16:     123.2 Tflops
mma_s8s8s32_32_8_16:      123.2 Tflops
mma_bf16bf16f32_16_16_16: 123.1 Tflops
mma_bf16bf16f32_32_8_16:  123.1 Tflops
mma_f16f16f16_16_16_16:   123.3 Tflops
mma_f16f16f16_32_8_16:    123.3 Tflops
mma_f16f16f32_16_16_16:   123.3 Tflops
mma_f16f16f32_32_8_16:    123.2 Tflops
mma_tf32tf32f32_16_16_8:  31.0 Tflops
```

## Compatibility

Tensor core operations that are not supported on your hardware will display "not supported".

## Architecture Support

- Volta: 70,72
- Turing: 75
- Ampere: 80,86,87
- Ada Lovelace: 89
- Hopper: 90,90a
- Blackwell : 100,110,120a

## License

This project is provided as-is.
