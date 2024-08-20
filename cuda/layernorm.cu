#include "layernorm.cuh"

// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void layernorm_forward_kernel1(float* x, float* mean, float* rstd,
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

__global__ void layernorm_forward_kernel2(const floatX* __restrict__ x, float* __restrict__ mean, float* __restrict__ rstd,
                                          const floatX* __restrict__ weight, const floatX* __restrict__ bias, floatX* __restrict__ y,
                                          int N, int H){
    
    int lane_Id = threadIdx.x % WARP_SIZE; // threadId inside a warp
    int warp_Id = threadIdx.x / WARP_SIZE; // warpId
    int num_warps = blockDim.x / WARP_SIZE; // number of warps

    int idx = blockIdx.x * num_warps + warp_Id;
    if(idx > N) {return;} // guard
    
    // the row of input that this group of threads is responsible for.
    const floatX* x_curr = x + idx*H;

    // mean [B x T of them, and each mean element is calculated by each warp]
    float sum = 0.f;
    for(int i=lane_id; i<H; i+=WARP_SIZE){
        sum += (float)x_curr[i];
    }
    sum = warpReduceSum(sum);
    float m = sum / H;
    if(lane_id==0 && mean != nullptr){
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.f;
    for(int i=lane_id; i<H; i+=WARP_SIZE){
        float diff = (float)x_curr[i] - m;
        sum += diff*diff;
    }
    sum = warpReduceSum(sum);
    float s = rsqrtf(sum/H + 1e-5f);
    if(lane_id==0 && rstd != nullptr){
        __stcs(rstd + idx, s);
    }

    
}

__global__ void layernorm_backward_kernel1(float* x, float* mean, float* rstd, float* weight,
                                           float* dx, float* dweight, float* dbias, float* dy,
                                           int N, int H){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        float curr_mean = mean[idx];
        float curr_rstd = rstd[idx];

        // Obtain derivatives of norm (n = s*(x - m) in forward pass)
        // to reduce ops
        float dnorm_mean = 0.f; // mean of dn
        float dnorm_norm_mean = 0.f; // mean of dn * n
        for(int h=0; h<H; ++h){
            float n = curr_rstd * (x[idx*H + h] - curr_mean);
            // since y = weight * n + bias, n' = (dn) = dy* weight
            float dn = dy[idx*H + h] * weight[h];
            dnorm_mean += dn;
            dnorm_norm_mean += dn * n;
        }
        dnorm_mean /= H;
        dnorm_norm_mean /= H;

        // Calculate the gradients
        for(int h=0; h<H; ++h){
            float n = curr_rstd * (x[idx*H + h] - curr_mean); 
            float dn = dy[idx*H + h] * weight[h];

            // bias derivative
            atomicAdd(&dbias[h], dy[idx*H + h]);
            // weight derivative
            atomicAdd(&dweight[h], dy[idx*H + h]*n);
            // x derivative
            // dx = dn * rstd + sum_over_H(dn * -rstd * 1/H) - (x - mean)*sum_over_H(dn * (x-mean)*rstd^3 / H)
            //    = rstd*(dn -1/H *sum_over_H(dn) - (x-mean)*sum_over_H(dn*(x-mean)*rstd^2 / H))
            //    = rstd*(dn -1/H *sum_over_H(dn) - (x-mean)*rstd*sum_over_H(dn*n/H))
            //    = rstd(dn - dnorm_mean - n * dnorm_norm_mean)
            float dval = 0.f;
            dval += dn; // term 1
            dval -= dnorm_mean; // term 2
            dval -= n * dnorm_norm_mean; // term 3
            dval *= curr_rstd; // scale
            dx[idx*H + h] += dval;
        }
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

void layernorm_backward1(float* x, float* mean, float* rstd, float* weight,
                         float* dx, float* dweight, float* dbias, float* dy,
                         int B, int T, int H, const int block_size){
    const int N = B*T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_backward_kernal1<<<grid_size, block_size>>>(x, mean, rstd, weight, dx, dweight, dbias, dy, N, H);
}