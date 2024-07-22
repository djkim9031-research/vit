#pragma once
#include <stddef.h>
#include <omp.h>

// 2D convolution, forward function.
// @param x             linearized input tensor
// @param kernel        linearized trainable kernel window (weights), shape: [OC, C, KH, KW]
// @param bias          linearized trainable kernel window (bias), shape [OC]
// @param y             linearized output tensors
// @param B             number of batches
// @param C             number of channels
// @param H             input height
// @param W             input width
// @param OC            number of output channels
// @param KH            kernel window height
// @param KW            kernel window width
// @param stride        (optional) kernel window stride (default: 1)
// @param padding       (optional) padding size to the input (default: 0)
//
inline void conv2d_forward(float* x, float* kernel, float* bias, float* y,
                           int B, int C, int H, int W, int OC, int KH, int KW,
                           int stride = 1, int padding = 0){
    
    // output height
    int OH = (H - KH + 2*padding) / stride + 1;
    // output width
    int OW = (W - KW + 2*padding) / stride + 1;

    #pragma omp parallel for collapse(4)
    for(int b=0; b<B; ++b){
        for(int oc=0; oc<OC; ++oc){
            for(int oh=0; oh<OH; ++oh){
                for(int ow=0; ow<OW; ++ow){
                    float val = (bias!=NULL)? bias[oc] : 0.f;
                    for(int ic=0; ic<C; ++ic){
                        for(int kh=0; kh<KH; ++kh){
                            for(int kw=0; kw<KW; ++kw){
                                int ih = oh*stride + kh - padding;
                                int iw = ow*stride + kw - padding; 
                                if(ih>=0 && ih<H && iw>=0 && iw<W){
                                    val += x[b*H*W*C + ic*H*W + ih*W + iw] * 
                                           kernel[oc*C*KH*KW + ic*KH*KW + kh*KW + kw];
                                }
                            }
                        }
                    }
                    y[b*OC*OH*OW + oc*OH*OW + oh*OW + ow] = val;
                }
            }
        }
    }
}

// 2D convolution, backward function.
// @param x             linearized input tensor
// @param kernel        linearized trainable kernel window (weights), shape: [OC, C, KH, KW]
// @param dx            linearized input derivatives
// @param dkernel       linearized trainable kernel window (weights) derivatives, shape: [OC, C, KH, KW]
// @param dbias         linearized trainable kernel window (bias) derivatives, shape [OC]
// @param dy            linearized output derivatives
// @param B             number of batches
// @param C             number of channels
// @param H             input height
// @param W             input width
// @param OC            number of output channels
// @param KH            kernel window height
// @param KW            kernel window width
// @param stride        (optional) kernel window stride (default: 1)
// @param padding       (optional) padding size to the input (default: 0)
//
inline void conv2d_backward(float* x, float* kernel, 
                            float* dx, float* dkernel, float* dbias, float* dy,
                            int B, int C, int H, int W, int OC, int KH, int KW,
                            int stride = 1, int padding = 0){
    
    // y = sum(kernel*x) + bias

    // output height
    int OH = (H - KH + 2*padding) / stride + 1;
    // output width
    int OW = (W - KW + 2*padding) / stride + 1;

    #pragma omp parallel for collapse(4)
    for(int b=0; b<B; ++b){
        for(int oc=0; oc<OC; ++oc){
            for(int oh=0; oh<OH; ++oh){
                for(int ow=0; ow<OW; ++ow){
                    // dL/db = dL/dy*dy/db = y'*1
                    float grad = dy[b*OC*OH*OW + oc*OH*OW + oh*OW + ow];
                    if(dbias!=NULL){
                        #pragma omp atomic
                        dbias[oc] += grad;
                    }
                    for(int ic=0; ic<C; ++ic){
                        for(int kh=0; kh<KH; ++kh){
                            for(int kw=0; kw<KW; ++kw){
                                int ih = oh*stride + kh - padding;
                                int iw = ow*stride + kw - padding; 
                                if(ih>=0 && ih<H && iw>=0 && iw<W){
                                    float curr_x = x[b*C*H*W + ic*H*W + ih*W + iw];
                                    // (elementwise) dL/dx = dL/dy*dy/dx = y'*kernel
                                    // (elementwise) dL/dkernel = y'*x
                                    #pragma omp atomic
                                    dkernel[oc*C*KH*KW + ic*KH*KW + kh*KW + kw] += curr_x*grad;
                                    if(dx != NULL){
                                        #pragma omp atomic
                                        dx[b*H*W*C + ic*H*W + ih*W + iw] += kernel[oc*C*KH*KW + ic*KH*KW + kh*KW + kw]*grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}