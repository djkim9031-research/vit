#include "embeddings.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void embeddings_forward_kernel1(float* x1, float* x2, float* pos_embd, float* y,
                                           int B, int P, int H, int OH, int OW){
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int p = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z * blockDim.z + threadIdx.z;

    if(b < B && p < (P+1) && h < H){
        float val = 0.f;
        if(p==0){
            val = x2[h];
        } else{
            // [b, h, oh, ow] => [b, oh*ow, h]
            val = x1[b*H*OH*OW + h*OH*OW + (p-1)];
        }

        // Concatenation of patch embedding with class token.
        // then, adding position embedding.
        val += pos_embd[p*H + h];
        y[b*(P+1)*H + p*H + h] = val;
    }
}

__global__ void embeddings_backward_kernel1(float* x1, float* dx1, float* dx2, 
                                            float* dpos_embd, float* dy,
                                            int B, int P, int H, int OH, int OW){
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int p = blockIdx.y * blockDim.y + threadIdx.y;
    int h = blockIdx.z * blockDim.z + threadIdx.z;

    if(b < B && p < (P+1) && h < H){
        float grad = dy[b*(P+1)*H + p*H + h];
        if(p==0){
            atomicAdd(&dx2[h], grad);
        } else{
            // [b, oh*ow, h] => [b, h, oh, ow]
            atomicAdd(&dx1[b*H*OH*OW + h*OH*OW + (p-1)], grad);
        }
        atomicAdd(&dpos_embd[p*H + h], grad);
    }
}


// -----------------------------------------------------------------------------------------
// kernel launcher

void embeddings_forward1(float* x1, float* x2, float* pos_embd, float* y,
                         int B, int P, int H, int OH, int OW, const int cubrt_block_size){
    
    dim3 gridDim(ceil_div(B, cubrt_block_size), ceil_div(P+1, cubrt_block_size), ceil_div(H, cubrt_block_size));
    dim3 blockDim(cubrt_block_size, cubrt_block_size, cubrt_block_size);

    embeddings_forward_kernel1<<<gridDim, blockDim>>>(x1, x2, pos_embd, y, B, P, H, OH, OW);
}

void embeddings_backward1(float* x1, float* dx1, float* dx2, 
                          float* dpos_embd, float* dy,
                          int B, int P, int H, int OH, int OW, const int cubrt_block_size){
    
    dim3 gridDim(ceil_div(B, cubrt_block_size), ceil_div(P+1, cubrt_block_size), ceil_div(H, cubrt_block_size));
    dim3 blockDim(cubrt_block_size, cubrt_block_size, cubrt_block_size);

    embeddings_backward_kernel1<<<gridDim, blockDim>>>(x1, dx1, dx2, dpos_embd, dy, B, P, H, OH, OW);
}