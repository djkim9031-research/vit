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