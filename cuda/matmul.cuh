#pragma once

#include <assert.h>
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
__global__ void matmul_forward_kernel1(floatX* x, floatX* y, floatX* weight, floatX* bias,
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
__global__ void matmul_backward_kernel1(floatX* x, floatX* weight, floatX* dx, floatX* dweight, floatX* dbias,
                                        floatX* dy, int B_in_r, int in_c, int ou_c);

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
void matmul_forward1(floatX* x, floatX* y, floatX* weight, floatX* bias,
                     int B, int in_r, int in_c, int ou_c, const int sqrt_block_size);


// Matmul backward kernel launcher 1
// @param x                     linearized input tensors [B*in_r, in_c]
// @param weight                linearized weight tensors [in_c, ou_c]
// @param dx                    linearized input tensor derivatives [B*in_r, in_c]
// @param dweight               linearized weight tensor derivatives [in_c, ou_c]
// @param dbias                 linearized bias tensor derivatives [ou_c]
// @param dy                    linearized output tensor derivatives [B*in_r, ou_c]
// @param B                     number of batches
// @param in_r                  input row dimensions
// @param in_c                  input col dimensions
// @param ou_c                  output col dimensions
// @param sqrt_block_size       sqrt of CUDA block size
//
void matmul_backward1(floatX* x, floatX* weight, floatX* dx, floatX* dweight, floatX* dbias,
                      floatX* dy, int B, int in_r, int in_c, int ou_c, const int sqrt_block_size);

// Modified matmul forward kernel launcher 1
// Modified matmul forward function, to be used prior to the classification stage of ViT.
// This is due to the fact that the attention block output should be sliced at first index
// of the sequence T, corresponding to the cls_token, which is responsible for classification.
//
// @param x                     linearized input tensors
// @param y                     linearized output tensors
// @param weight                linearized weight tensors
// @param bias                  linearized bias tensors
// @param B                     number of batches
// @param T                     sequence length (patch length + 1)
// @param H                     hidden dimension size
// @param NC                    number of classes
// @param t                     index t in the sequence T to be sliced 
// @param sqrt_block_size       sqrt of CUDA block size
//
void matmul_forward_with_slicing_at_t1(floatX* x, floatX* y, floatX* weight, floatX* bias,
                                       int B, int T, int H, int NC, int t, const int sqrt_block_size);

// Modified matmul forward kernel launcher 2
// Optimized one function call from the kernel
// Modified matmul forward function, to be used prior to the classification stage of ViT.
// This is due to the fact that the attention block output should be sliced at first index
// of the sequence T, corresponding to the cls_token, which is responsible for classification.
//
// @param x                     linearized input tensors
// @param y                     linearized output tensors
// @param weight                linearized weight tensors
// @param bias                  linearized bias tensors
// @param B                     number of batches
// @param T                     sequence length (patch length + 1)
// @param H                     hidden dimension size
// @param NC                    number of classes
// @param t                     index t in the sequence T to be sliced 
// @param sqrt_block_size       sqrt of CUDA block size
//
void matmul_forward_with_slicing_at_t2(floatX* x, floatX* y, floatX* weight, floatX* bias,
                                       int B, int T, int H, int NC, int t, const int sqrt_block_size);


// Modified matmul backward kernel launcher
// Modified matmul backward function, to be used after the backward call of crossentropy_softmax_backward.
//
// @param x                     linearized input tensors [B, T, H]
// @param weight                linearized weight tensors [H, NC]
// @param dx                    linearized input tensor derivatives [B, T, H]
// @param dweight               linearized weight tensor derivatives [H, NC]
// @param dbias                 linearized bias tensor derivatives [NC]
// @param dy                    linearized output tensor derivatives [B, 1, NC]
// @param B                     number of batches
// @param T                     sequence length (patch length + 1)
// @param H                     hidden dimension size
// @param NC                    number of classes
// @param t                     index t in the sequence T to be sliced 
// @param sqrt_block_size       sqrt of CUDA block size
// 
void matmul_backward_with_slicing_at_t(floatX* x, floatX* weight, floatX* dx, floatX* dweight, floatX* dbias,
                                       floatX* dy, int B, int T, int H, int NC, int t, const int sqrt_block_size);



// Matmul kernel launcher using cuBLAS to perform GEMM (or strided batched GEMM)
// GEMM performs d = alpha*op(a * b) + beta*op(c)
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
//
// @param a                     column-major linearized matrix of shape [m x k] if trans = False
// @param b                     column-major linearized matrix of shape [k x n] if trans = False
// @param bias                  bias tensor of shape [n]
// @param d                     column-major linearized output matrix of shape [m x n]
// @param m                     dimension parameter (row(`a`), row(`d`) if trans = False)
// @param n                     dimension parameter (col(`b`), col(`d`) if trans = False)
// @param k                     dimension parameter (col(`a`), row(`b`) if trans = False)
// @param stream                CUDA stream
// @param transA                Whether matrix `a` needs to be transposed
// @param transB                Whether matrix `b` needs to be transposed
// @param batch_count           number of batches for batched GEMM
// @param strideA               number of stride matrix `a `needs to make for strided batched GEMM to access the next `a` element in address
// @param strideB               number of stride matrix `b` needs to make for strided batched GEMM to access the next `b` element in address
// @param strideD               number of stride matrix `d` needs to make for strided batched GEMM to access the next `d` element in address
// @param accumulate            Whether to accumulate matrix (e.g., `d` += `c`)
// @param pre_gelu              column-major pre-gelu tensor (post GEMM) that feeds to gelu function (if in-function GELU is opted)
//                              If NULL, gelu is not calculated inside this function.
// @param backward              Whether the current function call is for forward-pass or backward-pass
//
void matmul_cublaslt(const floatX* a, const floatX* b, const floatX* bias, floatX* d, int m, int n, int k,
                     cudaStream_t stream=0, bool transA=false, bool transB=false, int batch_count=0,
                     size_t strideA=0, size_t strideB=0, size_t strideD=0, bool accumulate=0,
                     floatX* pre_gelu=NULL, bool backward=false);

// Matmul kernel launcher using cuBLAS to perform GEMM (or strided batched GEMM) - forward pass entry function.
//
// @param x                     linearized input tensors
// @param y                     linearized output tensors
// @param weight                linearized weight tensors
// @param bias                  linearized bias tensors
// @param B                     number of batches
// @param in_r                  input row dimensions
// @param in_c                  input col dimensions
// @param ou_c                  output col dimensions
// @param stream                CUDA stream
// @param pre_gelu              column-major pre-gelu tensor (post GEMM) that feeds to gelu function (if in-function GELU is opted)
//                              If NULL, gelu is not calculated inside this function.
//
void matmul_forward_cublaslt(floatX* x, floatX* y, floatX* weight, floatX* bias,
                             int B, int in_r, int in_c, int ou_c, cudaStream_t stream, floatX* pre_gelu);