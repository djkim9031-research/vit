#pragma once
#include <omp.h>
#include <stdlib.h>
#include <cstring>
#include <cassert>

// Helper function to flatten and transpose the input tensor
inline void flatten_and_transpose(float* x, float* x_mod, int B, int C, int H, int W){
    int HW = H*W;
    //#pragma omp parallel for collapse(2)
    for(int b=0; b<B; ++b){
        for(int c=0; c<C; ++c){
            for(int h=0; h<H; ++h){
                for(int w=0; w<W; ++w){
                    int src_idx = b*C*H*W + c*H*W + h*W + w;
                    int dst_idx = b*HW*C + (h*W + w)*C + c;
                    x_mod[dst_idx] = x[src_idx];
                }
            }
        }
    }
}

// Helper function to reverse the flatten and transpose
inline void reverse_flatten_and_transpose(float* x_mod, float* x, int B, int C, int H, int W){
    int HW = H*W;
    //#pragma omp parallel for collapse(2)
    for(int b=0; b<B; ++b){
        for(int c=0; c<C; ++c){
            for(int h=0; h<H; ++h){
                for(int w=0; w<W; ++w){
                    int src_idx = b*HW*C + (h*W + w)*C + c;
                    int dst_idx = b*C*H*W + c*H*W + h*W + w;
                    x[dst_idx] = x_mod[src_idx];
                }
            }
        }
    }
}


// Embedding, forward function
// @param x1            linearized patch embedding original input tensor, [batch_size, hidden_size, OH, OW]
// @param x2            linearized class token input tensor, [1, 1, hidden_size]
// @param pos_embd      linearized position embedding tensor, [1, num_patches+1, hidden_size]
// @param y             linearized output tensor, [batch_size, num_patches+1, hidden_size]
// @param B             number of batches
// @param P             number of patches
// @param H             hidden size
// @param OH            patch embedding's original tensor height dim
// @param OW            patch embedding's original tensor width dim          
//
inline void embeddings_forward(float* x1, float* x2, float* pos_embd, float* y,
                               int B, int P, int H, int OH, int OW){

    assert(OH*OW == P);
    float* x1_mod = (float*)malloc(B*H*OH*OW*sizeof(float));
    // x1_mod will become [batch_size, num_patches, hidden_size]
    flatten_and_transpose(x1, x1_mod, B, H, OH, OW);
    
    #pragma omp parallel for collapse(2)
    for(int b=0; b<B; ++b){
        for(int p=0; p<P+1; ++p){
            for(int h=0; h<H; ++h){
                float val = 0.f;
                if(p==0){
                    val = x2[h];
                } else {
                    val = x1_mod[b*P*H + (p-1)*H + h];
                }
                // concatenation of patch embedding with class token.
                // then, adding position embedding.
                val += pos_embd[p*H + h];
                y[b*(P+1)*H + p*H + h] = val;
            }
        }
    }

    free(x1_mod);
}

// Embedding, backward function
// @param x1            linearized patch embedding original input tensor, [batch_size, hidden_size, OH, OW]         
// @param dx1           linearized patch embedding original input derivatives
// @param dx2           linearized class token input derivatives
// @param dpos_embd     linearized position embedding derivatives
// @param dy            linearized output derivatives
// @param B             number of batches
// @param P             number of patches
// @param H             hidden size
// @param OH            patch embedding's original tensor height dim
// @param OW            patch embedding's original tensor width dim
//
inline void embeddings_backward(float* x1, float* dx1, float* dx2, 
                                float* dpos_embd, float* dy,
                                int B, int P, int H, int OH, int OW){

    assert(OH*OW == P);
    float* dx1_mod = (float*)malloc(B*H*OH*OW*sizeof(float));
    memset(dx1_mod, 0.0, B*H*OH*OW*sizeof(float));

    #pragma omp parallel for collapse(2)
    for(int b=0; b<B; ++b){
        for(int p=0; p<P+1; ++p){
            for(int h=0; h<H; ++h){
                float grad = dy[b*(P+1)*H + p*H + h];
                if(p==0){
                    #pragma omp atomic
                    dx2[h] += grad;
                } else {
                    #pragma omp atomic
                    dx1_mod[b*P*H + (p-1)*H + h] += grad;
                }
                #pragma omp atomic
                dpos_embd[p*H + h] += grad;
            }
        }
    }

    reverse_flatten_and_transpose(dx1_mod, dx1, B, H, OH, OW);
    free(dx1_mod);
}