#pragma once

#include "common.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

// 2D convolution, forward kernal function 1 (naive implementation)
//
// @param x                     linearized input tensor
// @param kernel                linearized trainable kernel window (weights), shape: [OC, C, KH, KW]
// @param bias                  linearized trainable kernel window (bias), shape [OC]
// @param y                     linearized output tensors
// @param B                     number of batches
// @param C                     number of channels
// @param H                     input height
// @param W                     input width
// @param OC                    number of output channels
// @param KH                    kernel window height
// @param KW                    kernel window width
// @param stride                kernel window stride 
// @param padding               padding size to the input 
//
__global__ void conv2d_forward_kernel1(float* x, float* kernel, float* bias, float* y,
                                       int B, int C, int H, int W, int OC, int KH, int KW,
                                       int stride, int padding);

// 2D convolution, backward kernal function 1 (naive implementation)
//
// @param x                     linearized input tensor
// @param kernel                linearized trainable kernel window (weights), shape: [OC, C, KH, KW]
// @param dx                    linearized input derivatives
// @param dkernel               linearized trainable kernel window (weights) derivatives, shape: [OC, C, KH, KW]
// @param dbias                 linearized trainable kernel window (bias) derivatives, shape [OC]
// @param dy                    linearized output derivatives
// @param B                     number of batches
// @param C                     number of channels
// @param H                     input height
// @param W                     input width
// @param OC                    number of output channels
// @param KH                    kernel window height
// @param KW                    kernel window width
// @param stride                kernel window stride 
// @param padding               padding size to the input 
//
__global__ void conv2d_backward_kernel1(float* x, float* kernel, 
                                        float* dx, float* dkernel, float* dbias, float* dy,
                                        int B, int C, int H, int W, int OC, int KH, int KW,
                                        int stride, int padding);


// -----------------------------------------------------------------------------------------
// kernel launcher

// 2D convolution forward kernal launcher 1
//
// @param x                     linearized input tensor
// @param kernel                linearized trainable kernel window (weights), shape: [OC, C, KH, KW]
// @param bias                  linearized trainable kernel window (bias), shape [OC]
// @param y                     linearized output tensors
// @param B                     number of batches
// @param C                     number of channels
// @param H                     input height
// @param W                     input width
// @param OC                    number of output channels
// @param KH                    kernel window height
// @param KW                    kernel window width
// @param stride                kernel window stride 
// @param padding               padding size to the input 
// @param cubrt_block_size      cubic root of CUDA block size (three dimension CUDA block is allocated per conv2d op.)
//
void conv2d_forward1(float* x, float* kernel, float* bias, float* y,
                     int B, int C, int H, int W, int OC, int KH, int KW,
                     int stride, int padding, const int cubrt_block_size);


// 2D convolution backward kernal launcher 1
//
// @param x                     linearized input tensor
// @param kernel                linearized trainable kernel window (weights), shape: [OC, C, KH, KW]
// @param dx                    linearized input derivatives
// @param dkernel               linearized trainable kernel window (weights) derivatives, shape: [OC, C, KH, KW]
// @param dbias                 linearized trainable kernel window (bias) derivatives, shape [OC]
// @param dy                    linearized output derivatives
// @param B                     number of batches
// @param C                     number of channels
// @param H                     input height
// @param W                     input width
// @param OC                    number of output channels
// @param KH                    kernel window height
// @param KW                    kernel window width
// @param stride                kernel window stride 
// @param padding               padding size to the input 
// @param cubrt_block_size      cubic root of CUDA block size (three dimension CUDA block is allocated per conv2d op.)
//
void conv2d_backward1(float* x, float* kernel, 
                      float* dx, float* dkernel, float* dbias, float* dy,
                      int B, int C, int H, int W, int OC, int KH, int KW,
                      int stride, int padding, const int cubrt_block_size);