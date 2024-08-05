#pragma once

#include "common.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

// Matrix multiplication, forward kernal function 1
// @param x                     linearized input tensors
// @param y                     linearized output tensors
// @param weight                linearized weight tensors
// @param bias                  linearized bias tensors
// @param B_in_r                number of batches * input row dimensions
// @param in_c                  input col dimensions
// @param ou_c                  output col dimensions
// With input tensor [B_in_r, in_c], the weight tensor is of
// [in_c, ou_c], and bias is of [ou_c].
// output tensor is, then, of [B_in_r, ou_c]
// y = x*w + b
//
__global__ void matmul_forward_kernel1(float* x, float* y, float* weight, float* bias,
                                       int B_in_r, int in_c, int ou_c);



// -----------------------------------------------------------------------------------------
// kernel launcher

// Matmul forward kernal launcher 1
//
// @param x                     linearized input tensors
// @param y                     linearized output tensors
// @param weight                linearized weight tensors
// @param bias                  linearized bias tensors
// @param B                     number of batches
// @param in_r                  input row dimensions
// @param in_c                  input col dimensions
// @param ou_c                  output col dimensions
// @param sqrt_block_size       sqrt of CUDA block size (two dimension CUDA block is allocated per matmul op.)
//
void matmul_forward1(float* x, float* y, float* weight, float* bias,
                     int B, int in_r, int in_c, int ou_c, const int sqrt_block_size);


