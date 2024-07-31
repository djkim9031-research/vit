#pragma once
#include <stddef.h>
#include <omp.h>

// Matrix multiplication, forward function.
// @param x             linearized input tensors
// @param y             linearized output tensors
// @param weight        linearized weight tensors
// @param bias          linearized bias tensors
// @param B             number of batches
// @param in_r          input row dimensions
// @param in_c          input col dimensions
// @param ou_c          output col dimensions
// With input tensor [B, in_r, in_c], the weight tensor is of
// [in_c, ou_c], and bias is of [ou_c].
// output tensor is, then, of [B, in_r, ou_c]
// y = x*w + b
//
inline void matmul_forward(float* x, float* y, float* weight, float* bias,
                           int B, int in_r, int in_c, int ou_c){
    //#pragma omp parallel for collapse(3)
    for(int b=0; b<B; ++b){
        for(int r=0; r<in_r; ++r){
            for(int oc=0; oc<ou_c; ++oc){
                float val = (bias != NULL) ? bias[oc] : 0.f;
                for(int ic=0; ic<in_c; ++ic){
                    val += x[b*in_r*in_c + r*in_c + ic] * weight[ic*ou_c + oc];
                }
                y[b*in_r*ou_c + r*ou_c + oc] = val;
            }
        }
    }
}

// Matrix multiplication, backward function.
// @param x             linearized input tensors [B, in_r, in_c]
// @param weight        linearized weight tensors [in_c, ou_c]
// @param dx            linearized input tensor derivatives [B, in_r, in_c]
// @param dweight       linearized weight tensor derivatives [in_c, ou_c]
// @param dbias         linearized bias tensor derivatives [ou_c]
// @param dy            linearized output tensor derivatives [B, in_r, ou_c]
// @param B             number of batches
// @param in_r          input row dimensions
// @param in_c          input col dimensions
// @param ou_c          output col dimensions
//
inline void matmul_backward(float* x, float* weight, float* dx, float* dweight, float* dbias,
                            float* dy, int B, int in_r, int in_c, int ou_c){
    
    //#pragma omp parallel for collapse(3)
    for(int b=0; b<B; ++b){
        for(int r=0; r<in_r; ++r){
            for(int oc=0; oc<ou_c; ++oc){
                float grad = dy[b*in_r*ou_c + r*ou_c + oc];
                if(dbias!=NULL){
                    //#pragma omp atomic
                    dbias[oc] += grad;
                }
                for(int ic=0; ic<in_c; ++ic){
                    dx[b*in_r*in_c + r*in_c + ic] += grad * weight[ic*ou_c + oc];
                    dweight[ic*ou_c + oc] += grad * x[b*in_r*in_c + r*in_c + ic];
                }

            }
        }
    }
}

// Helper function to extract a tensor at index t in sequence T of the original tensor.
// This is used for modified matmul function, which will be called before the classfication stage
// of ViT.
// 
// @param orig          linearized original input tensors [B, T, H]
// @param extracted     linearized extracted tensors, slice at index t [B, 1, H]
// @param B             number of batches
// @param T             sequence length (patch length + 1)
// @param H             hidden dimension size
// @param t             index t in the sequence T to be sliced
//
inline void slice_tensor_at_t(float* orig, float* extracted,
                              int B, int T, int H, int t){
    for(int b=0; b<B; ++b){
        for(int h=0; h<H; ++h){
            extracted[b*H + h] = orig[b*T*H + t*H + h]; 
        }
    }
}

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
//
inline void matmul_forward_with_slicing_at_t(float* x, float* y, float* weight, float* bias,
                                             int B, int T, int H, int NC, int t){
    // Allocate memory for the sliced tensor
    float* extracted_x = (float*)calloc(B*H, sizeof(float));
    // Extract at index t
    slice_tensor_at_t(x, extracted_x, B, T, H, t);

    // Perform matrix forward operation
    matmul_forward(extracted_x, y, weight, bias, B, 1, H, NC);

    // free up the temporary resource
    free(extracted_x);
}

// Modified matmul backward function, to be used after the backward call of crossentropy_softmax_backward.
//
// @param x             linearized input tensors [B, T, H]
// @param weight        linearized weight tensors [H, NC]
// @param dx            linearized input tensor derivatives [B, T, H]
// @param dweight       linearized weight tensor derivatives [H, NC]
// @param dbias         linearized bias tensor derivatives [NC]
// @param dy            linearized output tensor derivatives [B, 1, NC]
// @param B             number of batches
// @param T             sequence length (patch length + 1)
// @param H             hidden dimension size
// @param NC            number of classes
// @param t             index t in the sequence T to be sliced 
//
inline void matmul_backward_with_slicing_at_t(float* x, float* weight, float* dx, float* dweight, float* dbias,
                                              float* dy, int B, int T, int H, int NC, int t){
    
    // Allocate memory for the sliced tensor and its gradient
    float* extracted_x = (float*)calloc(B*H, sizeof(float));
    float* extracted_dx = (float*)calloc(B*H, sizeof(float));
    // Extract at index t
    slice_tensor_at_t(x, extracted_x, B, T, H, t);

    // Perform matrix backward operation for extracted tensor.
    // The gradients will be propagated only at index t
    matmul_backward(extracted_x, weight, extracted_dx, dweight, dbias, 
                    dy, B, 1, H, NC);

    // Accumulate the gradient in the original tensor x, dx
    for(int b=0; b<B; ++b){
        for(int h=0; h<H; ++h){
            //#pragma omp atomic
            dx[b*T*H + t*H + h] += extracted_dx[b*H + h];
        }
    }

    // free up the temporary resources
    free(extracted_x);
    free(extracted_dx);
}
