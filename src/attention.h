#pragma once
#include <stddef.h>
#include <omp.h>
#include <math.h>

// Attention forward function.
// The qkv values are obtained in the previous stage with matmul
// and passed to this function to calculate the attention score
//
// @param x             linearized qkv input tensors (batch_size B, sequence length T, 3*hidden_size = 3H)
// @param preattn       linearized pre-attention scores (B, number of heads NH, T, T)
// @param attn          linearized attention scores (B, NH, T, T)
// @param y             linearized output tensors (B, T, H)
// @param B             number of batches
// @param T             sequence length
// @param H             hidden size
// @param NH            number of heads
//
inline void attention_forward(float* x, float* preattn, float* attn, float* y,
                              int B, int T, int H, int NH){
    int HS = H / NH; // head size
    float scale = 1.0/sqrtf(HS);

    // Original qkv, each [B, T, H]
    // It is treated each as [B, T, NH * HS] to calculate an attention score for each head
    #pragma omp parallel for collapse(3)
    for(int b=0; b<B; ++b){
        for(int t1=0; t1<T; ++t1){
            for(int nh=0; nh<NH; ++nh){
                float* query = x + b*T*3*H + t1*3*H + nh*HS; // +0 for query, +H key, +2H value
                // Calculate q*k and maxval
                float maxval = -10000.0f;
                for(int t2=0; t2<T; ++t2){
                    float* key = x + b*T*3*H + t2*3*H + nh*HS + H;

                    //q*k
                    float curr_val = 0.f;
                    for(int i=0; i<HS; ++i){
                        curr_val += query[i] * key[i];
                    }
                    curr_val *= scale;
                    if(curr_val > maxval){
                        maxval = curr_val;
                    }

                    // prior to softmax, scaled weights
                    preattn[b*NH*T*T + nh*T*T + t1*T + t2] = curr_val;
                }

                // Calculate the exp and keep track of sum.
                // Calculated maxval is used only for numerical stability.
                float expsum = 0.f;
                for(int t2=0; t2<T; ++t2){
                    float curr_exp_v = expf(preattn[b*NH*T*T + nh*T*T + t1*T + t2] - maxval);
                    expsum += curr_exp_v;
                    attn[b*NH*T*T + nh*T*T + t1*T + t2] = curr_exp_v;
                }
                float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;

                // Softmax normalization
                for(int t2=0; t2<T; ++t2){
                    attn[b*NH*T*T + nh*T*T + t1*T + t2] *= expsum_inv;
                }

                // Calculate output tensor by dot(attn, v)
                for(int i=0; i<HS; ++i) {y[b*T*H + t1*H + nh*HS + i] = 0.f;} // initialization
                for(int t2=0; t2<T; ++t2){
                    float* value = x + b*T*3*H + t2*3*H + nh*HS + 2*H;
                    float corr_attn = attn[b*NH*T*T + nh*T*T + t1*T + t2];
                    for(int i=0; i<HS; ++i){
                        y[b*T*H + t1*H + nh*HS + i] += corr_attn * value[i];
                    }
                }
            }
        }
    }
}