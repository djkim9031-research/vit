#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <float.h>

#define WARP_SIZE 32U

// Calculate the min number of grid required given the total number of operations (dividend)
// and block size (divisor)
template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor){
    return (dividend + divisor - 1)/divisor;
}

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