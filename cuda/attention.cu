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
using cache_type_bwd = std::map<std::tuple<int, int, int, int>, std::shared_ptr<fe::graph::Graph>>;

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

// Caching/look up cache for attention forward pass
// HS = H / NH where H is hidden_size
// From input tensor of shape [B, T, 3H], QKV vector is obtained by
// [B, T, 3H] => [B, T, 3, NH, HS] => permutation => [B, NH, T, HS] x 3 (for Q, K, V)
// cuDNN can handle permutation directly without an external logic for permutation.
// 
//
// @param B                     number of batches
// @param NH                    number of heads
// @param T                     sequence length
// @param HS                    head size
// @param is_inference_only     bool for whether this is for inference_only or not
//
auto lookup_cache_or_build_graph_fwd(int B, int NH, int T, int HS, int is_inference_only){

    static cache_type_fwd user_maintained_cache_fwd;
    auto key = std::make_tuple(B, NH, T, HS, is_inference_only);

    // Cache lookup if it exists
    auto it = user_maintained_cache_fwd.find(key);
    if(it != user_maintained_cache_fwd.end()){
        return it->second;
    }

    // Graph build operation and create cache
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // Each Q, K, V is of shape [B, NH, T, HS]
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                                .set_dim({B, NH, T, HS})
                                .set_uid(Q_UID)
                                .set_stride({T*3*NH*HS, HS, 3*NH*HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                                .set_dim({B, NH, T, HS})
                                .set_uid(K_UID)
                                .set_stride({T*3*NH*HS, HS, 3*NH*HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                                .set_dim({B, NH, T, HS})
                                .set_uid(V_UID)
                                .set_stride({T*3*NH*HS, HS, 3*NH*HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_uid(Attn_scale_UID)
                                .set_is_pass_by_value(true)
                                .set_data_type(fe::DataType_t::FLOAT));

    // scale dot product attention options
    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(false);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(true).set_dim({B, NH, T, HS}).set_stride({T*NH*HS, HS, NH*HS, 1}).set_uid(O_UID);

    assert(stats == nullptr || is_inference_onl false);
    if(is_inference_only == false){
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, NH, T, 1})
                               .set_stride({T*NH, T, 1, 1})
                               .set_uid(Stats_UID);
    }
    cuDNNFECheck(graph->validate());

    // Build the operation graph and execution part (very slow)
    cuDNNFECheck(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    cuDNNFECheck(graph->check_support(cudnn_handle));
    cuDNNFECheck(graph->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    if(graph->get_workspace_size() > cudnn_workspace_size){
        if(cudnn_workspace_size > 0){
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    return graph;
}

// Caching/look up cache for attention backward pass
// HS = H / NH where H is hidden_size
// From input tensor of shape [B, T, 3H], QKV vector is obtained by
// [B, T, 3H] => [B, T, 3, NH, HS] => permutation => [B, NH, T, HS] x 3 (for Q, K, V)
// cuDNN can handle permutation directly without an external logic for permutation.
// 
//
// @param B                     number of batches
// @param NH                    number of heads
// @param T                     sequence length
// @param HS                    head size
//
auto lookup_cache_or_build_graph_bwd(int B, int NH, int T, int HS){
    
    static cache_type_bwd user_maintained_cache_bwd;
    auto key = std::make_tuple(B, NH, T, HS);

    // Cache lookup if it exists
    auto it = user_maintained_cache_bwd.find(key);
    if(it != user_maintained_cache_bwd.end()){
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // (B, N, 3, HS, HS) must come from input `x` (which means we also need to convert FloatX to FP16)
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                                .set_dim({B, NH, T, HS})
                                .set_uid(Q_UID)
                                .set_stride({T*3*NH*HS, HS, 3*NH*HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                                .set_dim({B, NH, T, HS})
                                .set_uid(K_UID)
                                .set_stride({T*3*NH*HS, HS, 3*NH*HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                                .set_dim({B, NH, T, HS})
                                .set_uid(V_UID)
                                .set_stride({T*3*NH*HS, HS, 3*NH*HS, 1}));
    auto O = graph->tensor(fe::graph::Tensor_attributes().set_name("O")
                                .set_dim({B, NH, T, HS})
                                .set_uid(O_UID)
                                .set_stride({T*NH*HS, HS, NH*HS, 1}));
    auto dO = graph->tensor(fe::graph::Tensor_attributes().set_name("dO")
                                .set_dim({B, NH, T, HS})
                                .set_uid(dO_UID)
                                .set_stride({T*NH*HS, HS, NH*HS, 1}));
    auto stats = graph->tensor(fe::graph::Tensor_attributes().set_name("stats")
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_uid(Attn_scale_UID)
                                .set_is_pass_by_value(true)
                                .set_data_type(fe::DataType_t::FLOAT));
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes().set_name("flash_attention_backward")
                                .set_deterministic_algorithm(true) // cuDNN_Frontend >= 1.5 (version req.)
                                .set_causal_mask(false)
                                .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({B, NH, T, HS}).set_stride({T*3*NH*HS, HS, 3*NH*HS, 1}).set_uid(dQ_UID);
    dK->set_output(true).set_dim({B, NH, T, HS}).set_stride({T*3*NH*HS, HS, 3*NH*HS, 1}).set_uid(dK_UID);
    dV->set_output(true).set_dim({B, NH, T, HS}).set_stride({T*3*NH*HS, HS, 3*NH*HS, 1}).set_uid(dV_UID);
    cuDNNFECheck(graph->validate());

    // Build the operation graph and execution part (very slow)
    cuDNNFECheck(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    cuDNNFECheck(graph->check_support(cudnn_handle));
    cuDNNFECheck(graph->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace.
    // By default, cuDNN uses up to 256 MiB of workspace, so we don't want to just allocate the maximum.
    if(graph->get_workspace_size() > cudnn_workspace_size){
        if(cudnn_workspace_size > 0){
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    user_maintained_cache_bwd.insert({key, graph});
    return graph;
    
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

