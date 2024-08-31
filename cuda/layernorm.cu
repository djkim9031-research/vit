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

    // finalize normalization and scaling by weight/bias
    floatX* y_curr = y + idx*H;
    for(int i=lane_id; i<H; i += WARP_SIZE){
        // load and store using the `cs` streaming hint to the compiler.
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        // .cs operator ref: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
        float n = s * ((float)__ldcs(x_curr + i) - m);
        __stcs(y_curr + i, (floatX)(n*(float)weight[i] + (float)bias[i]));
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

__global__ void __launch_bounds_(512, 2)
    layernorm_backward_kernel2(const floatX* x, const float* mean, const float* rstd, const floatX* weight,
                               floatX* dx, floatX* dweight, floatX* dbias, const floatX* dy,
                               int B, int T, int H){
    
    int BLOCK_SIZE = blockDim.x;
    int num_warps_per_block = BLOCK_SIZE / WARP_SIZE; // number of warps in a block
    extern __shared__ float shared[]; // dynamically allocated params (shared among threads in the same block)

    int warp_Id = threadIdx.x / WARP_SIZE; // warp index within a block
    int base_Id = blockIdx.x * num_warps_per_block + warp_Id;
    int lane_Id = threadIdx.x % WARP_SIZE; // thread index within a warp
    int num_warps_per_grid = gridDim.x * num_warps_per_block;
    int H_per_iteration = WARP_SIZE * x128::size; // how many elements in H vector is covered by SIMD and warp.
    int iterations_H = CEIL_DIV(H, H_per_iteration); // how many iterations are needed to cover the entire elements in H vector with SIMD and warp.

    // The first half of the shared memory is for bias, and the second is for weight
    size_t rounded_H = CEIL_DIV(H, (32 * x128::size)) * (32 * x128::size); // number of H elements rounded up to the nearest multiple of 32 x x128::size
    float* dbias_shared = shared;
    float* dweight_shared = shared + rounded_H;

    // init shared memory to zero
    for(int i=threadIdx.x * f128::size; i < rounded_H; i+=BLOCK_SIZE*f128::size){
        store128(dbias_shared + i, f128::zeros());
        store128(dweight_shared + i, f128::zeros());
    }
    __syncthreads();

    for(int bt=base_Id; bt<B*T; bt+=num_warps_per_grid) {  // b, t dim handled per warp
        const floatX* dy_bt = dy + bt*H;
        const floatX* x_bt = x + bt*H;
        floatX* dx_bt = dx + bt*H;

        // first: two reduce operations
        // Obtain derivatives of norm (n = s*(x - m) in forward pass)
        // to reduce ops
        float dnorm_mean = 0.f; // mean of dn
        float dnorm_norm_mean = 0.f; // mean of dn * n
        for(int h=lane_Id * x128::size; h<H; h+=WARP_SIZE*x128::size){ // h dim handled per thread in warp
            x128 dy128_h = load128(dy_bt + h);
            x128 x128_h = load128(x_bt + h);
            x128 weight128_h = load128(weight + h);

            const float curr_mean = mean[bt];
            const float curr_rstd = rstd[bt];

            for(int k=0; k<x128::size; ++k){
                // per data in SIMD, 
                float n = curr_rstd * ((float)x128_h[k]- curr_mean);
                // since y = weight * n + bias, n' = (dn) = dy* weight
                float dn = (float)dy128_h[k] * (float)weight128_h[k];
                dnorm_mean += dn;
                dnorm_norm_mean += dn * n;
            }
        }
        dnorm_mean = warpReduceSum(dnorm_mean)/H;
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean)/H;
    }

}

// -----------------------------------------------------------------------------------------
// kernel launcher

void layernorm_forward1(float* x, float* mean, float* rstd,
                        float* weight, float* bias, float* y,
                        int B, int T, int H, const int block_size){
    const int N = B*T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_forward_kernel1<<<grid_size, block_size>>>(x, mean, rstd, weight, bias, y, N, H);
}

void layernorm_backward1(float* x, float* mean, float* rstd, float* weight,
                         float* dx, float* dweight, float* dbias, float* dy,
                         int B, int T, int H, const int block_size){
    const int N = B*T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_backward_kernel1<<<grid_size, block_size>>>(x, mean, rstd, weight, dx, dweight, dbias, dy, N, H);
}