#pragma once
#include <math.h>

// softmax forward function.
//
// @param logits        linearized input logit tensors (batch_size B, 1, num_classes = NC)
// @param probs         linearized output probability tensors (B, 1, NC)
// @param B             number of batches
// @param NC            number of classes
//
inline void softmax_forward(float* logits, float* probs,
                            int B, int NC){
    for(int b=0; b<B; ++b){
        //maxval calculation for numerical stability for expf.
        float maxval = -10000.0f;
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

// crossentropy forward function
//
// @param probs         linearized input probability tensors (B, 1, NC)
// @param targets       linearized ground truth label tensors (B, 1, 1)
// @param losses        linearized output losses tensors (B, 1, 1)
// @param B             number of batches
// @param NC            number of classes
//
inline void crossentropy_forward(float* probs, int* targets, float* losses,
                                 int B, int NC){
    for(int b=0; b<B; ++b){
        // loss = -log(probs_pred)
        int cls_idx = targets[b];
        losses[b] = -logf(probs[b*NC + cls_idx]);
    }
}

// crossentropy and softmax backward function
//
// @param probs         linearized input probability tensors (B, 1, NC)
// @param targets       linearized ground truth label tensors (B, 1, 1)
// @param dlogits       linearized logit tensors derivatives (B, 1, NC)
// @param dlosses       linearized losses tensors derivatives (B, 1, 1)
// @param B             number of batches
// @param NC            number of classes
//
inline void crossentropy_softmax_backward(float* probs, int* targets,
                                          float* dlogits, float* dlosses,
                                          int B, int NC){
    for(int b=0; b<B; ++b){
        float dloss = dlosses[b];
        int cls_idx = targets[b];
        for(int i=0; i<NC; ++i){
            // dL/d(logits) = dL/d(prob) * d(probs)/d(logits)
            // dL/d(probs) = -1/prob, d(probs)/d(logits) = softmax(target_logit)*(1-softmax(target_logit)) if logit == target
            // otherwise, softmax(target_logit)*(0 - softmax(curr_logit))
            // where prob = softmax output for target logit, and dL/d(prob) = 0 for all other logits.
            // so, dL/d(logits) = -1/prob * (prob) * (indicator - curr_prob) =  curr_prob - indicator.
            float curr_cls_prob = probs[b*NC + i];
            float indicator = i == cls_idx ? 1.f : 0.f;
            dlogits[b*NC + i] += (curr_cls_prob - indicator)*dloss;
        }
    }
}