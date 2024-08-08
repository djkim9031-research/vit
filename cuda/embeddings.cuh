#pragma once

#include "common.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

// Embedding, forward kernal function 1
//
// @param x1                    linearized patch embedding original input tensor, [batch_size, hidden_size, OH, OW]
// @param x2                    linearized class token input tensor, [1, 1, hidden_size]
// @param pos_embd              linearized position embedding tensor, [1, num_patches+1, hidden_size]
// @param y                     linearized output tensor, [batch_size, num_patches+1, hidden_size]
// @param B                     number of batches
// @param P                     number of patches
// @param H                     hidden size
// @param OH                    patch embedding's original tensor height dim
// @param OW                    patch embedding's original tensor width dim          
//
__global__ void embeddings_forward_kernel1(float* x1, float* x2, float* pos_embd, float* y,
                                           int B, int P, int H, int OH, int OW);

// Embedding, backward kernal function 1
//
// @param x1                    linearized patch embedding original input tensor, [batch_size, hidden_size, OH, OW]         
// @param dx1                   linearized patch embedding original input derivatives
// @param dx2                   linearized class token input derivatives
// @param dpos_embd             linearized position embedding derivatives
// @param dy                    linearized output derivatives
// @param B                     number of batches
// @param P                     number of patches
// @param H                     hidden size
// @param OH                    patch embedding's original tensor height dim
// @param OW                    patch embedding's original tensor width dim
//
__global__ void embeddings_backward_kernel1(float* x1, float* dx1, float* dx2, 
                                            float* dpos_embd, float* dy,
                                            int B, int P, int H, int OH, int OW);


// -----------------------------------------------------------------------------------------
// kernel launcher

// Embedding forward kernal launcher 1
//
// @param x1                    linearized patch embedding original input tensor, [batch_size, hidden_size, OH, OW]
// @param x2                    linearized class token input tensor, [1, 1, hidden_size]
// @param pos_embd              linearized position embedding tensor, [1, num_patches+1, hidden_size]
// @param y                     linearized output tensor, [batch_size, num_patches+1, hidden_size]
// @param B                     number of batches
// @param P                     number of patches
// @param H                     hidden size
// @param OH                    patch embedding's original tensor height dim
// @param OW                    patch embedding's original tensor width dim   
// @param cubrt_block_size      cubic root of CUDA block size (three dimension CUDA block is allocated per embedding op.)
//
void embeddings_forward1(float* x1, float* x2, float* pos_embd, float* y,
                         int B, int P, int H, int OH, int OW, const int cubrt_block_size);


// Embedding backward kernal launcher 1
//
// @param x1                    linearized patch embedding original input tensor, [batch_size, hidden_size, OH, OW]         
// @param dx1                   linearized patch embedding original input derivatives
// @param dx2                   linearized class token input derivatives
// @param dpos_embd             linearized position embedding derivatives
// @param dy                    linearized output derivatives
// @param B                     number of batches
// @param P                     number of patches
// @param H                     hidden size
// @param OH                    patch embedding's original tensor height dim
// @param OW                    patch embedding's original tensor width dim
// @param cubrt_block_size      cubic root of CUDA block size (three dimension CUDA block is allocated per embedding op.)
//
void embeddings_backward1(float* x1, float* dx1, float* dx2, 
                          float* dpos_embd, float* dy,
                          int B, int P, int H, int OH, int OW, const int cubrt_block_size);
