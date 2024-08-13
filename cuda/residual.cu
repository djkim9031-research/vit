#include "residual.cuh"


// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void residual_forward_kernel1(floatX* x1, floatX* x2, floatX* y, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        y[i] = x1[i] + x2[i];
    }
}

__global__ void residual_backward_kernel1(floatX* dx1, floatX* dx2, floatX* dy, int N){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        // Applying chainrule: 1*dy
        atomicAdd(&dx1[i], dy[i]);
        atomicAdd(&dx2[i], dy[i]);
    }
}


// -----------------------------------------------------------------------------------------
// kernel launcher

void residual_forward1(floatX* x1, floatX* x2, floatX* y, int N, const int block_size){

    const int grid_size = ceil_div(N, block_size);
    residual_forward_kernel1<<<grid_size, block_size>>>(x1, x2, y, N);
}

void residual_backward1(floatX* dx1, floatX* dx2, floatX* dy, int N, const int block_size){

    const int grid_size = ceil_div(N, block_size);
    residual_backward_kernel1<<<grid_size, block_size>>>(dx1, dx2, dy, N);
}