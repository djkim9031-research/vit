#pragma once

#include <math.h>
#define M_PI 3.14159265358979323846
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

// Gelu nonlinearity with tanh approximation, foward fucntion.
// It is given by 0.5*x*(1+tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// @param x             linearized input tensors
// @param y             linearized output tensors
// @param N             number of elements
//
inline void gelu_forward(float* x, float* y, int N){
    for(int i=0; i<N; ++i){
        float x_i = x[i];
        float cube_term = 0.0044715f * x_i * x_i * x_i;
        y[i] = 0.5 * x_i * (1.f + tanhf(GELU_SCALING_FACTOR * (x_i + cube_term)));
    }
}

// Gelu nonlinaerity with tanh approximation, backward function.
// Given q = sqrt(2/pi)*(x + 0.044715*x^3),
// d(GELU(x))/dx = 0.5*(1+tanh(q)) + 0.5*x*((sech(q))^2)*sqrt(2/pi)*(1+0.134145*x^2)
// @param x             linearized input tensors
// @param dx            linearized input derivatives
// @param dy            linearized output derivatives
// @param N             number of elements
//
inline void gelu_backward(float* x, float* dx, float* dy, int N){
    for(int i=0; i<N; ++i){
        float x_i = x[i];
        float cube_term = 0.0044715f * x_i * x_i * x_i;
        float q = GELU_SCALING_FACTOR * (x_i + cube_term);
        float tanh_term = 0.5*(1 + tanhf(q));
        float cosh = coshf(q);
        float sech_term = 1.f/(cosh*cosh);
        float gelu_grad = tanh_term + 0.5f*x_i*sech_term*GELU_SCALING_FACTOR*(1.f+0.134145f*x_i*x_i);
        // Applying chain rule
        dx[i] += gelu_grad*dy[i];
    }
}