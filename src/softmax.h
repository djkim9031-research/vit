#pragma once
#include <omp.h>
#include <math.h>


// softmax forward function.
//
// @param logits        linearized input logit tensors (batch_size B, sequence length T, num_classes = NC)
// @param probs         linearized output probability tensors (B, T, NC)
// @param B             number of batches
// @param T             sequence length
// @param NC            number of classes
//
inline void softmax_forward(float* logits, float* probs,
                            int B, int T, int NC){
    #pragma omp parallel for collapse(2)
    for(int b=0; b<B; ++b){
        for(int t=0; t<T; ++t){
            //maxval calculation for numerical stability for expf.
            float maxval = -10000.0f;
            for(int i=0; i<NC; ++i){
                if(logits[b*T*NC + t*NC + i] > maxval){
                    maxval = logits[b*T*NC + t*NC + i];
                }
            }
            float sum = 0.f;
            for(int i=0; i<NC; ++i){
                probs[b*T*NC + t*NC + i] = expf(logits[b*T*NC + t*NC + i] - maxval);
                sum += probs[b*T*NC + t*NC + i];
            }
            for(int i=0; i<NC; ++i){
                probs[b*T*NC + t*NC + i] /= sum;
            }
        }
    }
}
