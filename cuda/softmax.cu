#include "softmax.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void softmax_forward_kernel1(float* logits, float* probs,
                                        int B, int NC){
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(b < B){

        // maxval calculation for numerical stability for expf.
        float maxval = -FLT_MAX;
        for(int i=0; i<NC; ++i){
            if(logits[b*NC + i] > maxval){
                maxval = logits[b*NC + i];
            }
        }
        float sum = 0.f;
        for(int i=0; i<NC; ++i){
            probs[b*NC + i] = expf(logits[b*NC + i] - maxval);
            sum += probs[b*NC + i];
        }
        for(int i=0; i<NC; ++i){
            probs[b*NC + i] /= sum;
        }
    }
}

// -----------------------------------------------------------------------------------------
// kernel launcher

void softmax_forward1(float* logits, float* probs,
                      int B, int NC, const int block_size){
    
    const int grid_size = ceil_div(N, block_size);
    softmax_forward_kernel1<<<grid_size, block_size>>>(logits, probs, B, NC);
}