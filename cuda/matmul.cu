#include "matmul.cuh"


// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void matmul_forward_kernel1(float* x, float* y, float* weight, float* bias,
                                       int B_in_r, int in_c, int ou_c){
    
    int B_in_r_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int ou_c_dim = blockIdx.y * blockDim.y + threadIdx.y;
    if(B_in_r_dim < B_in_r && ou_c_dim < ou_c){
        float val = (bias != NULL) ? bias[ou_c_dim] : 0.f;
        for(int ic = 0; ic<in_c; ++ic){
            val += x[B_in_r_dim*in_c + ic] * weight[ic*ou_c + ou_c_dim];
        }

        y[B_in_r_dim*ou_c + ou_c_dim] = val;
    }
}

__global__ void matmul_backward_kernel1(float* x, float* weight, float* dx, float* dweight, float* dbias,
                                        float* dy, int B_in_r, int in_c, int ou_c){
    
    int B_in_r_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int ou_c_dim = blockIdx.y * blockDim.y + threadIdx.y;

    if(B_in_r_dim < B_in_r && ou_c_dim < ou_c){
        float grad = dy[B_in_r_dim*ou_c + ou_c_dim];

        if(dbias != NULL){
            atomicAdd(&dbias[ou_c_dim], grad);
        }

        for(int ic = 0; ic < in_c; ++ic){
            atomicAdd(&dx[B_in_r_dim*in_c + ic], grad * weight[ic*ou_c + ou_c_dim]);
            atomicAdd(&dweight[ic*ou_c + ou_c_dim], grad * x[B_in_r_dim*in_c + ic]);
        }
    }
}

// -----------------------------------------------------------------------------------------
// kernel launcher

void matmul_forward1(float* x, float* y, float* weight, float* bias,
                     int B, int in_r, int in_c, int ou_c, const int sqrt_block_size){
    
    dim3 gridDim(ceil_div(B*in_r, sqrt_block_size), ceil_div(ou_c, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel1<<<gridDim, blockDim>>>(x, y, weight, bias, B*in_r, in_c, ou_c);
}

void matmul_backward1(float* x, float* weight, float* dx, float* dweight, float* dbias,
                      float* dy, int B, int in_r, int in_c, int ou_c, const int sqrt_block_size){
    
    dim3 gridDim(ceil_div(B*in_r, sqrt_block_size), ceil_div(ou_c, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_backward_kernel<<<gridDim, blockDim>>>(x, weight, dx, dweight, dbias, dy, B*in_r, in_c, ou_c);
}