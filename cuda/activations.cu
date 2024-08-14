#include "activations.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void gelu_forward_kernel1(floatX* x, floatX* y, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        floatX x_i = x[i];
        floatX cube_term = 0.044715f * x_i * x_i * x_i;
        y[i] = 0.5 * x_i * (1.f + tanhf(GELU_SCALING_FACTOR * (x_i + cube_term)));
    }
}

__global__ void gelu_forward_kernel2(floatX* x, floatX* y){
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_y;
    x128 packed_x = load128cs(x + i);
    for(int k=0; k<packed_x.size; ++k){
        float x_i = (float)packed_x[k];
        float cube_term = 0.044715f * x_i * x_i * x_i;
        packed_y[k] = (floatX)(0.5 * x_i * (1.f + tanhf(GELU_SCALING_FACTOR * (x_i + cube_term))));
    }
    store128(y+i, packed_y);
}

__global__ void gelu_backward_kernel1(floatX* x, floatX* dx, floatX* dy, int N){
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

__global__ void gelu_backward_kernel2(floatX* x, floatX* derivatives){
    int i = (blockIdx.x * blockDim.x + threadIdx.x)*x128::size;

    x128 packed_dx;
    x128 packed_x = load128cs(x + i);
    x128 packed_dy = load128(derivatives + i);
    for(int k=0; k<packed_x.size; ++k){
        float x_i = (float)packed_x[k];
        float cube_term = 0.044715f * x_i * x_i * x_i;
        float q = GELU_SCALING_FACTOR * (x_i + cube_term);
        float tanh_term = 0.5*(1 + tanhf(q));
        float cosh = coshf(q);
        float sech_term = 1.f/(cosh*cosh);
        float gelu_grad = tanh_term + 0.5f*x_i*sech_term*GELU_SCALING_FACTOR*(1.f+0.134145f*x_i*x_i);
        packed_dx[k] = (floatX)(gelu_grad*(float)packed_dy[k]);
    }
    store128(derivatives+i, packed_dx);
}

// -----------------------------------------------------------------------------------------
// kernel launcher

void gelu_forward1(floatX* x, floatX* y, int N, const int block_size){
    const int grid_size = ceil_div(N, block_size);
    gelu_forward_kernel1<<<grid_size, block_size>>>(x, y, N);
}

void gelu_forward2(floatX* x, floatX* y, int N, const int block_size, cudaStream_t stream){
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_forward_kernel2<<<grid_size, block_size, 0, stream>>>(x, y);
}

void gelu_backward1(floatX* x, floatX* dx, floatX* dy, int N, const int block_size){
    const int grid_size = ceil_div(N, block_size);
    gelu_backward_kernel1<<<grid_size, block_size>>>(x, dx, dy, N);
}

void gelu_backward2(floatX* x, floatX* derivatives, int N, const int block_size, cudaStream_t stream){
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_backward_kernel2<<<grid_size, block_size, 0, stream>>>(x, derivatives);
}

