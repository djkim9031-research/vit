#include "attention.cuh"


// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void attention_query_key_kernel1(float* x, float* preattn, 
                                            int B, int T, int H, int NH){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B*NH*T*T; // preattn shape [B, NH, T, T]

    int t2 = idx % T;
    int t1 = (idx/T)%T;
    int nh = (idx/(T*T))%NH;
    int b = idx/(NH*T*T);
    int HS = H / NH; // head size

    if (idx < total_threads){

        float* query = x + b*T*3*H + t1*3*H + nh*HS; // +0 for query
        float* key = x + b*T*3*H + t2*3*H + nh*HS + H; // +1 for key

        // q@k
        float curr_val = 0.f;
        for(int i=0; i<HS; ++i){
            curr_val += query[i] * key[i];
        }
        curr_val *= 1/sqrtf(HS);

        preattn[idx] = curr_val;
    }
}

__global__ void attention_softmax_kernel1(float* preattn, float* attn,
                                          int B, int T, int NH){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B*T*NH;

    int nh = idx % NH;
    int t1 = (idx/NH)%T;
    int b = idx/(NH*T);

    if(idx < total_threads){

        // find maxval
        float maxval = -FLT_MAX;
        for(int t2=0; t2<T; ++t2){
            if(preattn[b*NH*T*T + nh*T*T + t1*T + t2] > maxval){
                maxval = preattn[b*NH*T*T + nh*T*T + t1*T + t2];
            }
        }

        // Calculate the exp and keep track of sum.
        // Calculated maxval is used only for numerical stability.
        float expsum = 0.f;
        for(int t2=0; t2<T; ++t2){
            float curr_exp_v = expf(preattn[b*NH*T*T + nh*T*T + t1*T + t2] - maxval);
            expsum += curr_exp_v;
            attn[b*NH*T*T + nh*T*T + t1*T + t2] = curr_exp_v;
        }
        float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;

        // Softmax normalization
        for(int t2=0; t2<T; ++t2){
            attn[b*NH*T*T + nh*T*T + t1*T + t2] *= expsum_inv;
        }
    }
}

__global__ void attention_value_kernel1(float* x, float* attn, float* y,
                                        int B, int T, int H, int NH){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B*T*NH;

    int nh = idx % NH;
    int t1 = (idx/NH)%T;
    int b = idx/(NH*T);
    int HS = H / NH; // head size

    if(idx < total_threads){

        // Calculate output tensor by attn@v
        for(int i=0; i<HS; ++i) {y[b*T*H + t1*H + nh*HS + i] = 0.f;} // initialization
        for(int t2=0; t2<T; ++t2){
            const float* value = x + b*T*3*H + t2*3*H + nh*HS + 2*H; // +2 for value
            float corr_attn = attn[b*NH*T*T + nh*T*T + t1*T + t2];
            for(int i=0; i<HS; ++i){
                y[b*T*H + t1*H + nh*HS + i] += corr_attn * value[i];
            }
        }
    }
}


__global__ void attention_backward_kernel1(float* x, float* attn, 
                                           float* dx, float* dpreattn, float* dattn, float* dy,
                                           int B, int T, int H, int NH){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B*T*NH;

    int nh = idx % NH;
    int t1 = (idx/NH)%T;
    int b = idx/(NH*T);
    int HS = H / NH; // head size
    float scale = 1.0/sqrtf(HS);

    if(idx < total_threads){
        float* dquery = dx + b*T*3*H + t1*3*H + nh*HS;
        float* query = x + b*T*3*H + t1*3*H + nh*HS;

        // backward to get dvalue and dattn
        for(int t2=0; t2<T; ++t2){
            float* dvalue = dx + b*T*3*H + t2*3*H + nh*HS + 2*H;
            float* value = x + b*T*3*H + t2*3*H + nh*HS + 2*H;
            for(int i=0; i<HS; ++i){
                // y[i] = attn[i] * value[i]
                atomicAdd(&dattn[b*NH*T*T + nh*T*T + t1*T + t2], value[i] * dy[b*T*H + t1*H + nh*HS + i]);
                atomicAdd(&dvalue[i], attn[b*NH*T*T + nh*T*T + t1*T + t2] * dy[b*T*H + t1*H + nh*HS + i]);
            }
        }

        // backward pass through softmax
        // given input [x1, x2, ... xk], the partial derivative is
        // given as (d(softmax(xi))/d(xk) = ):
        // 1. if i == k, softmax(xi)*(1-softmax(xi))
        // 2. otherwise, -softmax(xi)*softmax(xk)
        for(int t2=0; t2<T; ++t2){
            for(int t3=0; t3<T; ++t3){
                float indicator = t2 == t3 ? 1.f : 0.f;
                float local_derivative = attn[b*NH*T*T + nh*T*T + t1*T + t2] * (indicator - attn[b*NH*T*T + nh*T*T + t1*T + t3]);
                atomicAdd(&dpreattn[b*NH*T*T + nh*T*T + t1*T + t3], local_derivative * dattn[b*NH*T*T + nh*T*T + t1*T + t2]);
            }
        }

        // backward through q@k
        for(int t2=0; t2<T; ++t2){
            float* dkey = dx + b*T*3*H + t2*3*H + nh*HS + H;
            float* key = x + b*T*3*H + t2*3*H + nh*HS + H;
            for(int i=0; i<HS; ++i){
                // preattn[i] = query[i]*key[i]*scale
                atomicAdd(&dquery[i], dpreattn[b*NH*T*T + nh*T*T + t1*T + t2] * scale * key[i]);
                atomicAdd(&dkey[i], dpreattn[b*NH*T*T + nh*T*T + t1*T + t2] * scale * query[i]);
            }
        }
    }
}


// -----------------------------------------------------------------------------------------
// kernel launcher

void attention_forward1(float* x, float* preattn, float* attn, float* y,
                        int B, int T, int H, int NH, const int block_size){
    
    int total_threads = B*NH*T*T;
    int num_blocks = ceil_div(total_threads, block_size);

    attention_query_key_kernel1<<<num_blocks, block_size>>>(x, preattn, B, T, H, NH);

    total_threads = B*T*NH;
    int num_blocks = ceil_div(total_threads, block_size);

    attention_softmax_kernel1<<<num_blocks, block_size>>>(preattn, attn, B, T, NH);
    attention_value_kernel1<<<num_blocks, block_size>>>(x, attn, y, B, T, H, NH);
}

void attention_backward1(float* x, float* attn, 
                         float* dx, float* dpreattn, float* dattn, float* dy,
                         int B, int T, int H, int NH){
    
    int total_threads = B*T*NH;
    int num_blocks = ceil_div(total_threads, block_size);

    attention_backward_kernel1<<<num_blocks, block_size>>>(x, attn, dx, dpreattn, dattn, dy, B, T, H, NH);
}