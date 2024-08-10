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

__global__ void crossentropy_forward_kernel1(const float* probs, const int* targets, float* losses,
                                             int B, int NC){
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(b < B){

        // loss = - log(probs_pred)
        int cls_idx = targets[b];
        losses[b] = -logf(probs[b*NC + cls_idx]);
    }
}

__global__ void crossentropy_softmax_backward_kernel1(const float* probs, const int* targets,
                                                      float* dlogits, const float* dlosses,
                                                      int B, int NC){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx/NC;
    int nc = idx%NC;

    if(idx < B*NC){
        float dloss = dlosses[b];
        float cls_idx = targets[b];

        // dL/d(logits) = dL/d(prob) * d(probs)/d(logits)
        // dL/d(probs) = -1/prob, d(probs)/d(logits) = softmax(target_logit)*(1-softmax(target_logit)) if logit == target
        // otherwise, softmax(target_logit)*(0 - softmax(curr_logit))
        // where prob = softmax output for target logit, and dL/d(prob) = 0 for all other logits.
        // so, dL/d(logits) = -1/prob * (prob) * (indicator - curr_prob) =  curr_prob - indicator.

        float curr_cls_prob = probs[b*NC + nc];
        float indicator = nc == cls_idx ? 1.f : 0.f;
        atomicAdd(&dlogits[b*NC + nc], (curr_cls_prob - indicator)*dloss);
    }
}

// -----------------------------------------------------------------------------------------
// kernel launcher

void softmax_forward1(float* logits, float* probs,
                      int B, int NC, const int block_size){
    
    const int grid_size = ceil_div(B, block_size);
    softmax_forward_kernel1<<<grid_size, block_size>>>(logits, probs, B, NC);
}

void crossentropy_forward1(const float* probs, const int* targets, float* losses,
                           int B, int NC, const int block_size){
    
    const int grid_size = ceil_div(B, block_size);
    crossentropy_forward_kernel1<<<grid_size, block_size>>>(probs, targets, losses, B, NC);
}

void crossentropy_softmax_backward1(const float* probs, const int* targets,
                                    float* dlogits, const float* dlosses,
                                    int B, int NC, const int block_size){
    
    const int grid_size = ceil_div(B*NC, block_size);
    crossentropy_softmax_backward_kernel1<<<grid_size, block_size>>>(probs, targets, dlogits, dlosses, B, NC);
}