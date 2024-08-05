#include "layernorm.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void layernorm_forward_kernal1(float* x, float* mean, float* rstd,
                                          float* weight, float* bias, float* y,
                                          int N, int H){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if(idx < N){
        // Calculate the mean over H-dim.
        float m = 0.f;
        for(int h=0; h<H; ++h){
            m += x[idx*H + h];
        }
        m /= H;

        // Calculate the variance (without Bessel's correction).
        float v = 0.f;
        for(int h=0; h<H; ++h){
            float xshift = x[idx*H + h] - m;
            v += xshift * xshift;
        }
        v /= H;

        // Calculate the rstd.
        float s = 1.f/sqrtf(v + eps);

        // Obtain the output y.
        for(int h=0; h<H; ++h){
            float n = (s*(x[idx*H + h] - m)); // normalize
            y[idx*H + h] = n * weight[h] + bias[h]; // scale and shift
        }

        // Cache the mean and rstd for the backward pass.
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// -----------------------------------------------------------------------------------------
// kernel launcher

void layernorm_forward1(float* x, float* mean, float* rstd,
                        float* weight, float* bias, float* y,
                        int B, int T, int H, const int block_size){
    const int N = B*T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_forward_kernal1<<<grid_size, block_size>>>(x, mean, rstd, weight, bias, y, N, H);
}