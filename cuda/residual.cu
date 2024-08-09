#include "residual.cuh"


// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void residual_forward_kernel1(float* x1, float* x2, float* y, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        y[i] = x1[i] + x2[i];
    }
}


// -----------------------------------------------------------------------------------------
// kernel launcher

void residual_forward1(float* x1, float* x2, float* y, int N, const int block_size){

    const int grid_size = ceil_div(N, block_size);
    residual_forward_kernel1<<<grid_size, block_size>>>(x1, x2, y, N);
}