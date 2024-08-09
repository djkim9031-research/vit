#pragma once

#include "common.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

// Softmax, forward kernal function 1
//
// @param logits        linearized input logit tensors (batch_size B, 1, num_classes = NC)
// @param probs         linearized output probability tensors (B, 1, NC)
// @param B             number of batches
// @param NC            number of classes
//
__global__ void softmax_forward_kernel1(float* logits, float* probs,
                                        int B, int NC);



// -----------------------------------------------------------------------------------------
// kernel launcher

// Softmax forward kernal launcher 1
//
// @param logits        linearized input logit tensors (batch_size B, 1, num_classes = NC)
// @param probs         linearized output probability tensors (B, 1, NC)
// @param B             number of batches
// @param NC            number of classes
// @param block_size    CUDA block size
//
void softmax_forward1(float* logits, float* probs,
                      int B, int NC, const int block_size);