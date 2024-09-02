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

    // Warp zero doesn't actually write to the _tmp_shared memory location, so we don't need to reserve memory here.
    // One solution is to change the addressing below to use (threadIdx.x - 32) as offset, but that causes register spills.
    // Instead, we mess with the base pointer here, which doesn't increase the register usage.
    // `_tmp_shared` is used only for warp_Id != 0 for accumulating partial results to in the block's shared memory, which warp_Id == 0 will load once finalized,
    // to calculate the final results and save to global memory.
    float* dbias_tmp_shared = shared + 2*rounded_H - WARP_SIZE*f128::size; 
    float* dweight_tmp_shared = shared + 2*rounded_H + f128::size * BLOCK_SIZE - 2*WARP_SIZE*f128::size; 

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

        for(int h=0; h<iterations_H; ++h){
            int global_h_idx = (lane_Id * x128::size) + (h * H_per_iteration); // per iteration covers warp_size * x128::size. 
                                                                               // This idx corresponds to an element this thread (lane_Id) in current warp at current h-th iteration.
                                                                               // The current thread contains SIMD vector
            
            x128 dy128_curr = x128::zeros();
            x128 x128_curr = x128::zeros();
            x128 dx128_curr = x128::zeros();
            x128 weight128_curr = x128::zeros();

            if(global_h_idx < H){
                dy128_curr = load128cs(dy_bt + global_h_idx);
                x128_curr = load128cs(x_bt + global_h_idx);
                dx128_curr = load128cs(dx_bt + global_h_idx);
                weight128_curr = load128cs(weight + global_h_idx);
            }

            for(int o=0; o<x128::size/f128::size; ++o){ // for half, 2 iterations, for single 1 iteration, for double 0 iteration. (double not supported)
                f128 dbias_f;
                f128 dweight_f;
                for(int i=0; i<f128::size; ++i){
                    int data_idx = o*f128::size + i;
                    float dy_i = (float)dy128_curr[data_idx];
                    float norm_bti = ((float)x128_curr[data_idx] - mean[bt])*rstd[bt];
                    dbias_f[i] = dy_i;
                    dweight_f[i] = dy_i * norm_bti;

                    // x derivative
                    // dx = dn * rstd + sum_over_H(dn * -rstd * 1/H) - (x - mean)*sum_over_H(dn * (x-mean)*rstd^3 / H)
                    //    = rstd*(dn -1/H *sum_over_H(dn) - (x-mean)*sum_over_H(dn*(x-mean)*rstd^2 / H))
                    //    = rstd*(dn -1/H *sum_over_H(dn) - (x-mean)*rstd*sum_over_H(dn*n/H))
                    //    = rstd(dn - dnorm_mean - n * dnorm_norm_mean)
                    float dval = 0.f;
                    dval += (float)weight128_curr[data_idx] * (float)dy128_curr[data_idx]; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd[bt]; // scale
                    dx128_curr[data_idx] = (floatX)((float)dx128_curr[data_idx] + dval);
                }

                // The idea is if warp_Id != 0, then results are stored in temporary shared memory (which is shared location with shared)
                // if warp_Id == 0, the results from temporary shared memory are first mapped to shared., then reduced summed to global memory.
                if(warp_Id != 0){
                    store128(dbias_tmp_shared + threadIdx.x * f128::size, dbias_f);
                    store128(dweight_tmp_shared + threadIdx.x * f128::size, dweight_f);
                }
                __syncthreads();
                if(warp_Id == 0){ // aggregating dbias and dweight results from warp_Id !=0 threads with dbias_f and dweight_f from warp_Id == 0.
                    for(int j=1; j<num_warps_per_block; ++j){
                        f128 dbias_tmp = load128(dbias_tmp_shared + f128::size*(threadIdx.x + j*WARP_SIZE));
                        f128 dweight_tmp = load128(dweight_tmp_shared + f128::size*(threadIdx.x + j*WARP_SIZE));
                        for(int i=0; i<f128::size; ++i){
                            dbias_f[i] += dbias_tmp[i];
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                __syncthreads();
                if(warp_Id == 0){
                    f128 db_old = load128(dbias_shared + global_h_idx + f128::size * o);
                    f128 dw_old = load128(dweight_shared + global_h_idx + f128::size * o);
                    for(int i=0; i<f128::size; ++i){
                        dbias_f[i] += db_old;
                        dweight_f[i] += dw_old;
                    }
                    store128(dbias_shared + global_h_idx + f128::size * o, dbias_f);
                    store128(dweight_shared + global_h_idx + f128::size * o, dweight_f);
                }
            }


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
    layernorm_forward_kernel1<<<grid_size, block_size>>>(x, mean, rstd, weight, bias, y, N, H);
}

void layernorm_backward1(float* x, float* mean, float* rstd, float* weight,
                         float* dx, float* dweight, float* dbias, float* dy,
                         int B, int T, int H, const int block_size){
    const int N = B*T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_backward_kernel1<<<grid_size, block_size>>>(x, mean, rstd, weight, dx, dweight, dbias, dy, N, H);
}