#pragma once

// Residual connection, forward function.
// @param x1            linearized input1 tensors
// @param x2            linearized input2 tensors
// @param y             linearized output tensors
// @param N             number of elements
//
inline void residual_forward(float* x1, float* x2, float* y, int N){
    for(int i=0; i<N; ++i){
        y[i] = x1[i] + x2[i];
    }
}

// Residual connection, forward function.
// y = x1 + x2 => dy/dx1 = 1, dy/dx2 = 1
// @param dx1           linearized input1 derivatives
// @param dx2           linearized input2 derivatives
// @param dy            linearized output derivatives
// @param N             number of elements
//
inline void residual_backward(float* dx1, float* dx2, float* dy, int N){
    for(int i=0; i<N; ++i){
        // Applying chainrule: 1*dy
        dx1[i] += dy[i];
        dx2[i] += dy[i];
    }
}