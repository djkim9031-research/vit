#pragma once

#include <assert.h>
#include "common.cuh"
#include "cuda_utils.cuh"

#define ENABLE_BF16
#define GELU_SCALING_FACTOR sqrtf(2.f/M_PI)

// -----------------------------------------------------------------------------------------
// CPU code reference


// Gelu nonlinearity with tanh approximation, foward fucntion.
// It is given by 0.5*x*(1+tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// @param x             linearized input tensors
// @param y             linearized output tensors
// @param N             number of elements
//
inline void gelu_forward(float* x, float* y, int N){
    for(int i=0; i<N; ++i){
        float x_i = x[i];
        float cube_term = 0.044715f * x_i * x_i * x_i;
        y[i] = 0.5 * x_i * (1.f + tanhf(GELU_SCALING_FACTOR * (x_i + cube_term)));
    }
}

// Gelu nonlinearity with tanh approximation, backward function.
// Given q = sqrt(2/pi)*(x + 0.044715*x^3),
// d(GELU(x))/dx = 0.5*(1+tanh(q)) + 0.5*x*((sech(q))^2)*sqrt(2/pi)*(1+0.134145*x^2)
// @param x             linearized input tensors
// @param dx            linearized input derivatives
// @param dy            linearized output derivatives
// @param N             number of elements
//
inline void gelu_backward(float* x, float* dx, float* dy, int N){
    for(int i=0; i<N; ++i){
        float x_i = x[i];
        float cube_term = 0.044715f * x_i * x_i * x_i;
        float q = GELU_SCALING_FACTOR * (x_i + cube_term);
        float tanh_term = 0.5*(1 + tanhf(q));
        float cosh = coshf(q);
        float sech_term = 1.f/(cosh*cosh);
        float gelu_grad = tanh_term + 0.5f*x_i*sech_term*GELU_SCALING_FACTOR*(1.f+0.134145f*x_i*x_i);
        // Applying chain rule
        dx[i] += gelu_grad*dy[i];
    }
}

// -----------------------------------------------------------------------------------------
// GPU kernels

// Gelu forward kernel function 1
//
// @param x             linearized input tensors
// @param y             linearized output tensors
// @param N             number of elements
//
__global__ void gelu_forward_kernel1(floatX* x, floatX* y, int N);

// Gelu forward kernel function 2 - SIMD
//
// @param x             linearized input tensors
// @param y             linearized output tensors
//
__global__ void gelu_forward_kernel2(floatX* x, floatX* y);

// Gelu backward kernel function 1
//
// @param x             linearized input tensors
// @param dx            linearized input derivatives
// @param dy            linearized output derivatives
// @param N             number of elements
//
__global__ void gelu_backward_kernel1(floatX* x, floatX* dx, floatX* dy, int N);

// Gelu backward kernel function 2 - SIMD
//
// @param x             linearized input tensors
// @param derivatives   linearized derivatives [inplace op]
//
__global__ void gelu_backward_kernel2(floatX* x, floatX* derivatives);

// -----------------------------------------------------------------------------------------
// kernel launcher

// Gelu forward kernel launcher 1
//
// @param x             linearized input tensors
// @param y             linearized output tensors
// @param N             number of elements
// @param block_size    CUDA block size
//
void gelu_forward1(floatX* x, floatX* y, int N, const int block_size);

// Gelu forward kernel launcher 2 - SIMD
//
// @param x             linearized input tensors
// @param y             linearized output tensors
// @param N             number of elements
// @param block_size    CUDA block size
// @param stream        CUDA stream
//
void gelu_forward2(floatX* x, floatX* y, int N, const int block_size, cudaStream_t stream);

// Gelu backward kernel launcher 1
//
// @param x             linearized input tensors
// @param dx            linearized input derivatives
// @param dy            linearized output derivatives
// @param N             number of elements
// @param block_size    CUDA block size
//
void gelu_backward1(floatX* x, floatX* dx, floatX* dy, int N, const int block_size);

// Gelu backward kernel launcher 2 - SIMD
//
// @param x             linearized input tensors
// @param derivatives   linearized derivatives [inplace op]
// @param N             number of elements
// @param block_size    CUDA block size
// @param stream        CUDA stream
//
void gelu_backward2(floatX* x, floatX* derivatives, int N, const int block_size, cudaStream_t stream);