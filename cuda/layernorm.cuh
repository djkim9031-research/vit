#pragma once

#include "common.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

// Layernorm function, forward kernal function 1
// Layer normalization over the given input tensor
// H dimension vector of activations gets normalized, then scaled and shifted.
//
// @param x             linearized input tensors (batch_size B, sequence length T, hidden_size = H)
// @param mean          linearized mean tensors over the last dimension (hidden size dim) [B, T]
// @param rstd          linearized reciprocal standard deviation tensors (B, T)
// @param weight        linearized weight(scale) tensor parameters (H)
// @param bias          linearized bias(shift) tensor parameters (H)
// @param y             linearized output tensors (B, T, H)
// @param N             number of batches(B) x sequence length (T)
// @param H             hidden size
// 
__global__ void layernorm_forward_kernal1(float* x, float* mean, float* rstd,
                                          float* weight, float* bias, float* y,
                                          int N, int H);

// Layernorm function, backward kernal function 1
//
// @param x             linearized input tensors (batch_size B, sequence length T, hidden_size = H)
// @param mean          linearized mean tensors over the last dimension (hidden size dim) [B, T]
// @param rstd          linearized reciprocal standard deviation tensors (B, T)
// @param weight        linearized weight(scale) tensor parameters (H)
// @param dx            linearized input tensor derivatives
// @param dweight       linearized weight tensor derivatives
// @param dbias         linearized bias tensor derivatives
// @param dy            linearized output tensor derivatives
// @param N             number of batches(B) x sequence length (T)
// @param H             hidden size
//
__global__ void layernorm_backward_kernal1(float* x, float* mean, float* rstd, float* weight,
                                           float* dx, float* dweight, float* dbias, float* dy,
                                           int N, int H);

// -----------------------------------------------------------------------------------------
// kernel launcher

// Layernorm forward kernel launcher 1
//
// @param x             linearized input tensors (batch_size B, sequence length T, hidden_size = H)
// @param mean          linearized mean tensors over the last dimension (hidden size dim) [B, T]
// @param rstd          linearized reciprocal standard deviation tensors (B, T)
// @param weight        linearized weight(scale) tensor parameters (H)
// @param bias          linearized bias(shift) tensor parameters (H)
// @param y             linearized output tensors (B, T, H)
// @param B             number of batches
// @param T             sequence length
// @param H             hidden size
// @param block_size    CUDA block size
// 
void layernorm_forward1(float* x, float* mean, float* rstd,
                        float* weight, float* bias, float* y,
                        int B, int T, int H, const int block_size);

// Layernorm backward kernel launcher 1
//
// @param x             linearized input tensors (batch_size B, sequence length T, hidden_size = H)
// @param mean          linearized mean tensors over the last dimension (hidden size dim) [B, T]
// @param rstd          linearized reciprocal standard deviation tensors (B, T)
// @param weight        linearized weight(scale) tensor parameters (H)
// @param dx            linearized input tensor derivatives
// @param dweight       linearized weight tensor derivatives
// @param dbias         linearized bias tensor derivatives
// @param dy            linearized output tensor derivatives
// @param B             number of batches
// @param T             sequence length
// @param H             hidden size
// @param block_size    CUDA block size
//
void layernorm_backward1(float* x, float* mean, float* rstd, float* weight,
                         float* dx, float* dweight, float* dbias, float* dy,
                         int B, int T, int H, const int block_size);
