#pragma once
#include <math.h>
#include <stdlib.h>
#include <string.h>

// This function initializes given trainable parameters to 0.
//
// @param params                Linearized trainable parameters.
// @param size                  The entire size of the trainable parameter.
//                              e.g., L1 x L2 x L3 tensor's size should be L1*L2*L3
//
inline void zeros_init(float* params, size_t size){
    memset(params, 0, size*sizeof(float));
}

// This function initializes given trainable parameters to 1.
//
// @param params                Linearized trainable parameters.
// @param size                  The entire size of the trainable parameter.
//                              e.g., L1 x L2 x L3 tensor's size should be L1*L2*L3
//
inline void ones_init(float* params, size_t size){
    for(size_t i=0; i<size; ++i){
        params[i] = 1.0f;
    }
}

// This function initializes given trainable parameters to random values from normal distribution.
// The gaussian random number is generated using Box-Muller method.
// Ref: https://www.tspi.at/2021/08/17/gaussianrng.html
//
// @param params                Linearized trainable parameters.
// @param size                  The entire size of the trainable parameter.
//                              e.g., L1 x L2 x L3 tensor's size should be L1*L2*L3
// @param mean                  The mean of the normal distribution.
// @param std                   The standard deviation of the normal distribution.
// @param seed                  Random seed for reproduciblity
//
inline void normal_init(float* params, size_t size, float mean, float std, unsigned int seed){
    srand(seed);
    for(size_t i=0; i<size; ++i){
        float u = ((float) rand()/RAND_MAX);
        float v = ((float) rand()/RAND_MAX);
        float z = sqrt(-2.0*log(u)) * cos(2.0 * M_PI * v);
        params[i] = mean + std*z;
    }
}