#include "attention.cuh"

// -----------------------------------------------------------------------------------------
// CUDNN frontend utils

#define NOMINMAX
#include <unistd.h>
#include <cudnn_frontend.h>
#include <map>

namespace fe = cudnn_frontend;
#if defined(ENABLE_FP32)
static assert(false, "cuDNN is not supported in FP32 mode.");
#elif defined(ENABLE_FP16)
#define CUDNN_16BIT fe::DataType_t::HALF
#else // Default to bfloat16
#define CUDNN_16BIT fe::DataType_t::HALF // testing with FP16 atm.
#endif

static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256 MiB)
static void* cudnn_workspace = NULL;

static void cudnn_check(cudnnStatus_t error, const char* file, int line){
    if(error != CUDNN_STATUS_SUCCESS){
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define cuDNNCheck(err) (cudnn_check(err, __FILE__, __LINE__))

static void cudnnFE_check(const fe::error_object& e, const char* file, int line){
    if(!e.is_good()){
        printf("[CUDNN FE ERROR] at file %s:%d\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);
    }
}
#define cuDNNFECheck(err) (cudnnFE_check(err, __FILE__, __LINE__))

enum UIDs{
    Q_UID,
    K_UID,
    V_UID,
    Attn_scale_UID,
    O_UID,
    Stats_UID,
    dO_UID,
    dQ_UID,
    dK_UID,
    dV_UID
};

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::map<std::tuple<int, int, int, int, int>, std::shared_ptr<fe::graph::Graph>>;
using cahce_type_bwd = std::map<std::tuple<int, int, int, int>, std::shared_ptr<fe::graph::Graph>>;

// Create cudnn handle (to be called at initialization)
void create_cudnn(){
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

// Destroy cudnn handle (to be released at the end)
void destroy_cudnn(){
    if(cudnn_workspace != NULL){
        cudaCheck(cudaFree(cudnn_workspace));
    }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}

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

