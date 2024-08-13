#include "activations.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void gelu_forward_kernal1(floatX* x, floatX* y, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        floatX x_i = x[i];
        floatX cube_term = 0.044715f * x_i * x_i * x_i;
        y[i] = 0.5 * x_i * (1.f + tanhf(GELU_SCALING_FACTOR * (x_i + cube_term)));
    }
}

__global__ void gelu_backward_kernal1(floatX* x, floatX* dx, floatX* dy, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        floatX x_i = x[i];
        floatX cube_term = 0.044715f * x_i * x_i * x_i;
        floatX q = GELU_SCALING_FACTOR * (x_i + cube_term);
        floatX tanh_term = 0.5*(1 + tanhf(q));
        floatX cosh = coshf(q);
        floatX sech_term = 1.f/(cosh*cosh);
        floatX gelu_grad = tanh_term + 0.5f*x_i*sech_term*GELU_SCALING_FACTOR*(1.f+0.134145f*x_i*x_i);
        // Applying chain rule
        dx[i] += gelu_grad*dy[i];
    }
}

// -----------------------------------------------------------------------------------------
// kernel launcher

void gelu_forward1(floatX* x, floatX* y, int N, const int block_size){
    const int grid_size = ceil_div(N, block_size);
    gelu_forward_kernal1<<<grid_size, block_size>>>(x, y, N);
}

void gelu_backward1(floatX* x, floatX* dx, floatX* dy, int N, const int block_size){
    const int grid_size = ceil_div(N, block_size);
    gelu_backward_kernal1<<<grid_size, block_size>>>(x, dx, dy, N);
}

