#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <float.h>

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <string>

#define WARP_SIZE 32U

// Calculate the min number of grid required given the total number of operations (dividend)
// and block size (divisor)
template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor){
    return (dividend + divisor - 1)/divisor;
}

// Simple macro for SIMD operations
#define CEIL_DIV(M, N) ((M+N-1)/N)

// ----------------------------------------------------------------------------
// Reduced/Mixed precision utilities

// precision datatype definition
#if defined(ENABLE_BF16)
typedef __nv_bfloat16 floatX;
typedef __nv_bfloat16 floatN;

#elif defined(ENABLE_FP16)
typedef half floatX;
typedef half floatN;

#else
typedef float floatX;
typedef float floatN;

#endif

// ----------------------------------------------------------------------------
// DType support

// enumerator to indentify the datatype of a tensor.
enum class DType : uint8_t {
    FP32, 
    FP16, 
    BF16
};

// Given a datatype enum, returns the underlying number of bytes
// for a scalar of that type
inline size_t sizeof_dtype(DType type) {
    switch (type) {
        case DType::FP32:
            return sizeof(float);
        case DType::FP16:
            return sizeof(half);
        case DType::BF16:
            return sizeof(nv_bfloat16);
        default: // handle or get compiler warning
            fprintf(stderr, "Unknown datatype\n");
            exit(EXIT_FAILURE);
    }
}

inline DType dtype_of(float* f) { return DType::FP32; }
inline DType dtype_of(half * f) { return DType::FP16; }
inline DType dtype_of(nv_bfloat16 * f) { return DType::BF16; }


// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
inline void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))


// ----------------------------------------------------------------------------
// Nsys profiler utils

class NvtxRange{
    public:
        NvtxRange(const char* s) {nvtxRangePush(s);}
        NvtxRange(const std::string& base_str, int number){
            std::string range_string = base_str + " " + std::to_string(number);
            nvtxRangePush(range_string.c_str());
        }
        ~NvtxRange() {nvtxRangePop();}
};
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)