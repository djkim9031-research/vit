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

// Helper function to extract a tensor at index t in sequence T of the original tensor.
// This is used for modified matmul function, which will be called before the classfication stage
// of ViT.
// 
// @param orig          linearized original input tensors [B, T, H]
// @param extracted     linearized extracted tensors, slice at index t [B, 1, H]
// @param B             number of batches
// @param T             sequence length (patch length + 1)
// @param H             hidden dimension size
// @param t             index t in the sequence T to be sliced
//
__global__ void slice_tensor_at_t_kernel(float* orig, float* extracted,
                                         int B, int T, int H, int t){
    
    int b_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int h_dim = blockIdx.y * blockDim.y + threadIdx.y;
    if(b_dim < B && h_dim < H){
        extracted[b_dim*H + h_dim] = orig[b_dim*T*H + t*H + h_dim];
    }
    __syncthreads();
}

// Matmul forward function with slicing at t.
//
// (B, T, H) => slice at index t => (B, 1, H) * weight(H, NC) + bias(NC) => (B, 1, NC)
//
// @param x             linearized input tensors
// @param y             linearized output tensors
// @param weight        linearized weight tensors
// @param bias          linearized bias tensors
// @param B             number of batches
// @param T             sequence length (patch length + 1)
// @param H             hidden dimension size
// @param NC            number of classes
// @param t             index t in the sequence T to be sliced 
//
__global__ void matmul_forward_with_slicing_at_t_kernel(float* x, float* y, float* weight, float* bias,
                                                        int B, int T, int H, int NC, int t){
    
    int b_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int nc_dim = blockIdx.y * blockDim.y + threadIdx.y;
    if(b_dim < B && nc_dim < NC){
        float val = (bias != NULL) ? bias[nc_dim] : 0.f;
        for(int h=0; h<H; ++h){
            float extracted_x = x[b_dim*T*H + t*H + h];
            val += extracted_x * weight[h*NC + nc_dim];
        }

        y[b_dim*NC + nc_dim] = val;
    }
}

// Matmul backward function with slicing at t.
//
// @param x             linearized input tensors [B, T, H]
// @param weight        linearized weight tensors [H, NC]
// @param dx            linearized input tensor derivatives [B, T, H]
// @param dweight       linearized weight tensor derivatives [H, NC]
// @param dbias         linearized bias tensor derivatives [NC]
// @param dy            linearized output tensor derivatives [B, 1, NC]
// @param B             number of batches
// @param T             sequence length (patch length + 1)
// @param H             hidden dimension size
// @param NC            number of classes
// @param t             index t in the sequence T to be sliced 
//
__global__ void matmul_backward_with_slicing_at_t_kernel(float* x, float* weight, float* dx, float* dweight, float* dbias,
                                                         float* dy, int B, int T, int H, int NC, int t){
    
    int b_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int nc_dim = blockIdx.y * blockDim.y + threadIdx.y;
    if(b_dim < B && nc_dim < NC){
        float grad = dy[b_dim*NC + nc_dim];

        if(dbias!=NULL){
            atomicAdd(&dbias[nc_dim], grad);
        }
        for(int h=0; h<H; ++h){
            float extracted_x = x[b_dim*T*H + t*H + h];

            atomicAdd(&dweight[h*NC + nc_dim], grad*extracted_x);
            atomicAdd(&dx[b_dim*T*H + t*H + h], grad*weight[h*NC + nc_dim]);
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

void matmul_forward_with_slicing_at_t1(float* x, float* y, float* weight, float* bias,
                                       int B, int T, int H, int NC, int t, const int sqrt_block_size){
    
    float* extracted_x;
    cudaMalloc(&extracted_x, B*H*sizeof(float));

    dim3 slice_gridDim(ceil_div(B, sqrt_block_size), ceil_div(H, sqrt_block_size));
    dim3 matmul_gridDim(ceil_div(B, sqrt_block_size), ceil_div(NC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);

    slice_tensor_at_t_kernel<<<slice_gridDim, blockDim>>>(x, extracted_x, B, T, H, t);
    matmul_forward_kernel1<<<matmul_gridDim, blockDim>>>(extracted_x, y, weight, bias, B, H, NC);

    cudaDeviceSynchronoize();
    cudaFree(extracted_x);
}

void matmul_forward_with_slicing_at_t2(float* x, float* y, float* weight, float* bias,
                                       int B, int T, int H, int NC, int t, const int sqrt_block_size){
    
    dim3 gridDim(ceil_div(B, sqrt_block_size), ceil_div(NC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);

    matmul_forward_with_slicing_at_t_kernel<<<gridDim, blockDim>>>(x, y, weight, bias, B, T, H, NC, t);

}

void matmul_backward_with_slicing_at_t(float* x, float* weight, float* dx, float* dweight, float* dbias,
                                       float* dy, int B, int T, int H, int NC, int t, const int sqrt_block_size){
    
    dim3 gridDim(ceil_div(B, sqrt_block_size), ceil_div(NC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);

    matmul_backward_with_slicing_at_t_kernel<<<gridDim, blockDim>>>(x, weight, dx, dweight, dbias, dy, B, T, H, NC, t);

}