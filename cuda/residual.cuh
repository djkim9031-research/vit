#pragma once

#include "common.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

// Residual connection, forward kernal function 1
//
// @param x1            linearized input1 tensors
// @param x2            linearized input2 tensors
// @param y             linearized output tensors
// @param N             number of elements
//
__global__ void residual_forward_kernel1(float* x1, float* x2, float* y, int N);


// Residual connection, backward kernal function 1
//
// y = x1 + x2 => dy/dx1 = 1, dy/dx2 = 1
// @param dx1           linearized input1 derivatives
// @param dx2           linearized input2 derivatives
// @param dy            linearized output derivatives
// @param N             number of elements
//
__global__ void residual_backward_kernel1(float* dx1, float* dx2, float* dy, int N);


// -----------------------------------------------------------------------------------------
// kernel launcher

// Residual connection forward kernal launcher 1
//
// @param x1            linearized input1 tensors
// @param x2            linearized input2 tensors
// @param y             linearized output tensors
// @param N             number of elements
// @param block_size    CUDA block size
//
void residual_forward1(float* x1, float* x2, float* y, int N, const int block_size);

// Residual connection backward kernal launcher 1
//
// y = x1 + x2 => dy/dx1 = 1, dy/dx2 = 1
// @param dx1           linearized input1 derivatives
// @param dx2           linearized input2 derivatives
// @param dy            linearized output derivatives
// @param N             number of elements
// @param block_size    CUDA block size
//
void residual_backward1(float* dx1, float* dx2, float* dy, int N, const int block_size);