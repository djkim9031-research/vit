#include "activations.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void gelu_forward_kernal1(float* x, float* y, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        float x_i = x[i];
        float cube_term = 0.044715f * x_i * x_i * x_i;
        y[i] = 0.5 * x_i * (1.f + tanhf(GELU_SCALING_FACTOR * (x_i + cube_term)));
    }
}

// -----------------------------------------------------------------------------------------
// kernel launcher

void gelu_forward1(float* x, float* y, int N, const int block_size){
    const int grid_size = ceil_div(N, block_size);
    gelu_forward_kernal1<<<grid_size, block_size>>>(x, y, N);
}

