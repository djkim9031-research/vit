#include "conv2d.cuh"


// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void conv2d_forward_kernel1(float* x, float* kernel, float* bias, float* y,
                                       int B, int C, int H, int W, int OC, int KH, int KW,
                                       int stride, int padding){
    
    // output height
    int OH = (H - KH + 2*padding) / stride + 1;
    // output width
    int OW = (W - KW + 2*padding) / stride + 1;

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    int oh = blockIdx.z * blockDim.z + threadIdx.z;

    if(b<B && oc < OC && oh < OH){
        for(int ow=0; ow<OW; ++ow){
            float val = (bias!=NULL)? bias[oc] : 0.f;
            for(int ic=0; ic<C; ++ic){
                for(int kh=0; kh<KH; ++kh){
                    for(int kw=0; kw<KW; ++kw){
                        int ih = oh*stride + kh - padding;
                        int iw = ow*stride + kw - padding; 
                        if(ih>=0 && ih<H && iw>=0 && iw<W){
                            val += x[b*H*W*C + ic*H*W + ih*W + iw] * 
                                   kernel[oc*C*KH*KW + ic*KH*KW + kh*KW + kw];
                        }
                    }
                }
            }
            y[b*OC*OH*OW + oc*OH*OW + oh*OW + ow] = val;
        }
    }
}

// -----------------------------------------------------------------------------------------
// kernel launcher

void conv2d_forward1(float* x, float* kernel, float* bias, float* y,
                     int B, int C, int H, int W, int OC, int KH, int KW,
                     int stride, int padding, const int cubrt_block_size){
    
    int OH = (H - KH + 2*padding) / stride + 1;

    dim3 gridDim(ceil_div(B, cubrt_block_size), ceil_div(OC, cubrt_block_size), ceil_div(OH, cubrt_block_size));
    dim3 blockDim(cubrt_block_size, cubrt_block_size, cubrt_block_size);

    conv2d_forward_kernel1<<<gridDim, blockDim>>>(x, kernel, bias, y, B, C, H, W, OC, KH, KW, stride, padding);
}