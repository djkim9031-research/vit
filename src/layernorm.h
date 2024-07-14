#pragma once
#include <math.h>

// Layernorm forward function.
// Layer normalization over the given input tensor
// H dimension vector of activations gets normalized, then scaled and shifted.
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
//
inline void layernorm_forward(float* x, float* mean, float* rstd,
                              float* weight, float* bias, float* y,
                              int B, int T, int H){
    float eps = 1e-5f;
    for(int b=0; b<B; ++b){
        for(int t=0; t<T; ++t){
            // Calculate the mean over H-dim
            float m=0.f;
            for(int h=0; h<H; ++h){
                m += x[b*T*H + t*H + h];
            }
            m /= H;

            // Calculate the variance (without Bessel's correction)
            float v = 0.f;
            for(int h=0; h<H; ++h){
                float xshift = x[b*T*H + t*H + h] - m;
                v += xshift*xshift;
            }
            v /= H;

            // Calculate the rstd
            float s = 1.f/sqrtf(v + eps);

            // Obtain the output y
            for(int h=0; h<H; ++h){
                float n = (s*(x[b*T*H + t*H + h] - m)); // normalize
                y[b*T*H + t*H + h] = n * weight[h] + bias[h]; // scale and shift
            }

            // cache the mean and rstd for the backward pass 
            mean[b*T + t] = m;
            rstd[b*T + t] = s;
        }
    }
}

// Layernorm backward function.
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
//
inline void layernorm_backward(float* x, float* mean, float* rstd, float* weight,
                               float* dx, float* dweight, float* dbias, float* dy,
                               int B, int T, int H){
    for(int b=0; b<B; ++b){
        for(int t=0; t<T; ++t){
            float curr_mean = mean[b*T + t];
            float curr_rstd = rstd[b*T + t];

            // Obtain derivatives of norm (n = s*(x - m) in forward pass)
            // to reduce ops
            float dnorm_mean = 0.f; // mean of dn
            float dnorm_norm_mean = 0.f; // mean of dn * n
            for(int h=0; h<H; ++h){
                float n = curr_rstd * (x[b*T*H + t*H + h] - curr_mean);
                // since y = weight * n + bias, n' = (dn) = dy* weight
                float dn = dy[b*T*H + t*H + h] * weight[h];
                dnorm_mean += dn;
                dnorm_norm_mean += dn * n;
            }
            dnorm_mean /= H;
            dnorm_norm_mean /= H;

            // Calculate the gradients
            for(int h=0; h<H; ++h){
                float n = curr_rstd * (x[b*T*H + t*H + h] - curr_mean); 
                float dn = dy[b*T*H + t*H + h] * weight[h];

                // bias derivative
                dbias[h] += dy[b*T*H + t*H + h];
                // weight derivative
                dweight[h] += dy[b*T*H + t*H + h] * n;
                // x derivative
                // dx = dn * rstd + sum_over_H(dn * -rstd * 1/H) - (x - mean)*sum_over_H(dn * (x-mean)*rstd^3 / H)
                //    = rstd*(dn -1/H *sum_over_H(dn) - (x-mean)*sum_over_H(dn*(x-mean)*rstd^2 / H))
                //    = rstd*(dn -1/H *sum_over_H(dn) - (x-mean)*rstd*sum_over_H(dn*n/H))
                //    = rstd(dn - dnorm_mean - n * dnorm_norm_mean)
                float dval = 0.f;
                dval += dn; // term 1
                dval -= dnorm_mean; // term 2
                dval -= n * dnorm_norm_mean; // term 3
                dval *= curr_rstd; // scale
                dx[b*T*H + t*H + h] += dval;
            }
        }
    }
}