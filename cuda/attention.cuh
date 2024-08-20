#pragma once

#include "common.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

// The first three functions are naive CUDA implementations.

// Attention forward kernal function 1
// preattention score calculation forward (q@k)
//
// @param x                     linearized qkv input tensors (batch_size B, sequence length T, 3*hidden_size = 3H)
// @param preattn               linearized pre-attention scores (B, number of heads NH, T, T)
// @param B                     number of batches
// @param T                     sequence length
// @param H                     hidden size
// @param NH                    number of heads
//
__global__ void attention_query_key_kernel1(float* x, float* preattn, 
                                            int B, int T, int H, int NH);

// Attention forward kernal function 1
// attention score softmax normalization forward (softmax(q@k))
//
// @param preattn               linearized pre-attention scores (B, number of heads NH, T, T)
// @param attn                  linearized attention scores (B, NH, T, T)
// @param B                     number of batches
// @param T                     sequence length
// @param NH                    number of heads
//
__global__ void attention_softmax_kernel1(float* preattn, float* attn,
                                          int B, int T, int NH);

// Attention forward kernal function 1
// attention score calculation forward (softmax(q@k) * value)
//
// @param x                     linearized qkv input tensors (batch_size B, sequence length T, 3*hidden_size = 3H)
// @param attn                  linearized attention scores (B, NH, T, T)
// @param y                     linearized output tensors (B, T, H)
// @param B                     number of batches
// @param T                     sequence length
// @param H                     hidden size
// @param NH                    number of heads
//
__global__ void attention_value_kernel1(float* x, float* attn, float* y,
                                        int B, int T, int H, int NH);


// Attention backward kernal function 1
// This function is a naive CUDA implementation.
// The qkv values are obtained in the previous stage with matmul
// and passed to this function to calculate the attention score
//
// @param x                     linearized qkv input tensors (batch_size B, sequence length T, 3*hidden_size = 3H)
// @param attn                  linearized attention scores (B, NH, T, T)
// @param dx                    linearized qkv input tensor derivatives
// @param dpreattn              linearized preattn derivatives
// @param dattn                 linearized attn derivatives
// @param dy                    linearized output derivatives
// @param B                     number of batches
// @param T                     sequence length
// @param H                     hidden size
// @param NH                    number of heads
//
__global__ void attention_backward_kernel1(float* x, float* attn, 
                                           float* dx, float* dpreattn, float* dattn, float* dy,
                                           int B, int T, int H, int NH);

// -----------------------------------------------------------------------------------------
// kernel launcher

// Attention forward kernal launcher 1
//
// @param x                     linearized qkv input tensors (batch_size B, sequence length T, 3*hidden_size = 3H)
// @param preattn               linearized pre-attention scores (B, number of heads NH, T, T)
// @param attn                  linearized attention scores (B, NH, T, T)
// @param y                     linearized output tensors (B, T, H)
// @param B                     number of batches
// @param T                     sequence length
// @param H                     hidden size
// @param NH                    number of heads
// @param block_size            CUDA block size
//
void attention_forward1(float* x, float* preattn, float* attn, float* y,
                        int B, int T, int H, int NH, const int block_size);


// Attention backward kernal launcher 1
//
// @param x                     linearized qkv input tensors (batch_size B, sequence length T, 3*hidden_size = 3H)
// @param attn                  linearized attention scores (B, NH, T, T)
// @param dx                    linearized qkv input tensor derivatives
// @param dpreattn              linearized preattn derivatives
// @param dattn                 linearized attn derivatives
// @param dy                    linearized output derivatives
// @param B                     number of batches
// @param T                     sequence length
// @param H                     hidden size
// @param NH                    number of heads
// @param block_size            CUDA block size
//
void attention_backward1(float* x, float* attn, 
                         float* dx, float* dpreattn, float* dattn, float* dy,
                         int B, int T, int H, int NH);

// -----------------------------------------------------------------------------------------
// CUDNN frontend utils

// cuDNN handle creation
//
void create_cudnn();

// cuDNN handle destruction
//
void destroy_cudnn();