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


// Matrix multiplication, backward kernel function 1
// @param x             linearized input tensors [B*in_r, in_c]
// @param weight        linearized weight tensors [in_c, ou_c]
// @param dx            linearized input tensor derivatives [B*in_r, in_c]
// @param dweight       linearized weight tensor derivatives [in_c, ou_c]
// @param dbias         linearized bias tensor derivatives [ou_c]
// @param dy            linearized output tensor derivatives [B*in_r, ou_c]
// @param B_in_r        number of batches * input row dimensions
// @param in_c          input col dimensions
// @param ou_c          output col dimensions
//
__global__ void matmul_backward_kernel1(float* x, float* weight, float* dx, float* dweight, float* dbias,
                                        float* dy, int B_in_r, int in_c, int ou_c);

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


// Matmul backward kernel launcher 1
// @param x             linearized input tensors [B*in_r, in_c]
// @param weight        linearized weight tensors [in_c, ou_c]
// @param dx            linearized input tensor derivatives [B*in_r, in_c]
// @param dweight       linearized weight tensor derivatives [in_c, ou_c]
// @param dbias         linearized bias tensor derivatives [ou_c]
// @param dy            linearized output tensor derivatives [B*in_r, ou_c]
// @param B             number of batches
// @param in_r          input row dimensions
// @param in_c          input col dimensions
// @param ou_c          output col dimensions
// @param sqrt_block_size  sqrt of CUDA block size
//
void matmul_backward1(float* x, float* weight, float* dx, float* dweight, float* dbias,
                      float* dy, int B, int in_r, int in_c, int ou_c, const int sqrt_block_size);

// Modified matmul forward kernel launcher 1
// Modified matmul forward function, to be used prior to the classification stage of ViT.
// This is due to the fact that the attention block output should be sliced at first index
// of the sequence T, corresponding to the cls_token, which is responsible for classification.
//
// @param x             linearized input tensors
// @param y             linearized output tensors
// @param weight        linearized weight tensors
// @param bias          linearized bias tensors
// @param B             number of batches
// @param T             sequence length (patch length + 1)
// @param H             hidden dimension size
// @param NC            number of classes
// @param t             index t in the sequence T to be sliced 
// @param sqrt_block_size  sqrt of CUDA block size
//
void matmul_forward_with_slicing_at_t1(float* x, float* y, float* weight, float* bias,
                                       int B, int T, int H, int NC, int t, const int sqrt_block_size);

// Modified matmul forward kernel launcher 2
// Optimized one function call from the kernel
// Modified matmul forward function, to be used prior to the classification stage of ViT.
// This is due to the fact that the attention block output should be sliced at first index
// of the sequence T, corresponding to the cls_token, which is responsible for classification.
//
// @param x             linearized input tensors
// @param y             linearized output tensors
// @param weight        linearized weight tensors
// @param bias          linearized bias tensors
// @param B             number of batches
// @param T             sequence length (patch length + 1)
// @param H             hidden dimension size
// @param NC            number of classes
// @param t             index t in the sequence T to be sliced 
// @param sqrt_block_size  sqrt of CUDA block size
//
void matmul_forward_with_slicing_at_t2(float* x, float* y, float* weight, float* bias,
                                       int B, int T, int H, int NC, int t, const int sqrt_block_size);