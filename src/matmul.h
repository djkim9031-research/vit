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
    #pragma omp parallel for collapse(3)
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
    
    #pragma omp parallel for collapse(3)
    for(int b=0; b<B; ++b){
        for(int r=0; r<in_r; ++r){
            for(int oc=0; oc<ou_c; ++oc){
                float grad = dy[b*in_r*ou_c + r*ou_c + oc];
                if(dbias!=NULL){
                    #pragma omp atomic
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