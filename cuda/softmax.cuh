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


// crossentropy, forward kernel function 1
//
// @param probs         linearized input probability tensors (B, 1, NC)
// @param targets       linearized ground truth label tensors (B, 1, 1)
// @param losses        linearized output losses tensors (B, 1, 1)
// @param B             number of batches
// @param NC            number of classes
//
__global__ void crossentropy_forward_kernel1(const float* probs, const int* targets, float* losses,
                                             int B, int NC);


// crossentropy and softmax backward kernel function 1
//
// @param probs         linearized input probability tensors (B, 1, NC)
// @param targets       linearized ground truth label tensors (B, 1, 1)
// @param dlogits       linearized logit tensors derivatives (B, 1, NC)
// @param dlosses       linearized losses tensors derivatives (B, 1, 1)
// @param B             number of batches
// @param NC            number of classes
//
__global__ void crossentropy_softmax_backward_kernel1(const float* probs, const int* targets,
                                                      float* dlogits, const float* dlosses,
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


// crossentropy forward kernel launcher 1
//
// @param probs         linearized input probability tensors (B, 1, NC)
// @param targets       linearized ground truth label tensors (B, 1, 1)
// @param losses        linearized output losses tensors (B, 1, 1)
// @param B             number of batches
// @param NC            number of classes
// @param block_size    CUDA block size
//
void crossentropy_forward1(const float* probs, const int* targets, float* losses,
                           int B, int NC, const int block_size);


// crossentropy and softmax backward kernel launcher 1
//
// @param probs         linearized input probability tensors (B, 1, NC)
// @param targets       linearized ground truth label tensors (B, 1, 1)
// @param dlogits       linearized logit tensors derivatives (B, 1, NC)
// @param dlosses       linearized losses tensors derivatives (B, 1, 1)
// @param B             number of batches
// @param NC            number of classes
// @param block_size    CUDA block size
//
void crossentropy_softmax_backward1(const float* probs, const int* targets,
                                    float* dlogits, const float* dlosses,
                                    int B, int NC, const int block_size);