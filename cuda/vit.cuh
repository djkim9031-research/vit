#pragma once
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/stat.h>

// ----------- Layer implementations in CUDA -----------
#include "conv2d.cuh"
#include "embeddings.cuh"
#include "layernorm.cuh"
#include "matmul.cuh"
#include "attention.cuh"
#include "residual.cuh"
#include "activations.cuh"
#include "softmax.cuh"


// global vars containing information about the GPU this process is running on
cudaDeviceProp deviceProp; 
cudaStream_t main_stream;

// ----------------------------------------------------------------------------
// ViT model definition

typedef struct{
    int image_width; // 32
    int image_height; // 32
    int channels; // 3
    int patch_size; // 4, therefore, for a 32 x 32 image, 8 x 8 patches are generated.
    int hidden_size; // 48
    int num_attention_heads; // 4, so head size = 48/4 = 12
    int num_layers; // L = 4
    int num_classes; // 10, CIFAR10 dataset
} ViTConfig;

// ----------------------------------------------------------------------------
// ViT model parameter tensors

constexpr const int NUM_PARAMETER_TENSORS=18;
typedef struct{
    floatX* patch_embd_kernel; // (hidden_size (H), num_channel(C), patch_height (PH), patch_width (PW))
    floatX* patch_embd_bias; // (H)
    floatX* cls_token; // (1, 1, H)
    floatX* pos_embd; // (1, 1 + num_patches (1+NP), H)
    floatX* ln1w; // (num_layers (L), H)
    floatX* ln1b; // (L, H)
    floatX* qkvw; // qkv projection matmul weight, (L, H, 3*H = 3*head_size*num_heads)
    floatX* qkvb; // (L, 3*H)
    floatX* attn_projw; // post attn matmul weight, (L, H, H)
    floatX* attn_projb; // (L, H)
    floatX* ln2w; // (L, H)
    floatX* ln2b; // (L, H)
    floatX* mlpw; // (L, H, 4*H)
    floatX* mlpb; // (L, 4*H)
    floatX* mlp_projw; // (L, 4*H, H)
    floatX* mlp_projb; // (L, H)
    floatX* clsw; // classifier matmul (H, num_classes (NC) )
    floatX* clsb; // (NC)
} ParameterTensors;
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS*sizeof(void*), "Inconsistent parameter tensor size.");

inline void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, ViTConfig* config){
    size_t nC = config->channels;
    size_t nL = config->num_layers;
    size_t H = config->hidden_size;
    size_t iW = config->image_width;
    size_t iH = config->image_height;
    size_t ps = config->patch_size;
    size_t nCls = config->num_classes;

    assert(iW%ps==0 && iH%ps==0); // Sanity check.
    size_t nP = (iW/ps)*(iH/ps); // number of patches.
    size_t T = nP + 1; // sequence length (+1 correspond to cls_token)

    // Define the param sizes
    param_sizes[0] = H*nC*nP; // patch_embd_kernel
    param_sizes[1] = H; // patch_embd_bias
    param_sizes[2] = H; // cls_token
    param_sizes[3] = T*H; // pos_embd
    param_sizes[4] = nL*H; // ln1w
    param_sizes[5] = nL*H; // ln1b
    param_sizes[6] = nL*H*3*H; // qkvw
    param_sizes[7] = nL*3*H; // qkvb
    param_sizes[8] = nL*H*H; // attn_projw
    param_sizes[9] = nL*H; // attn_projb
    param_sizes[10] = nL*H; // ln2w
    param_sizes[11] = nL*H; // ln2b
    param_sizes[12] = nL*H*4*H; // mlpw
    param_sizes[13] = nL*4*H; // mlpb
    param_sizes[14] = nL*4*H*H; // mlp_projw
    param_sizes[15] = nL*H; // mlp_projb
    param_sizes[16] = H*nCls; // clsw
    param_sizes[17] = nCls; // clsb

    // populate the parameter datatype sizes in bytes
    for(int i=0; i<NUM_PARAMETER_TENSORS; ++i){
        param_sizeof[i] = sizeof(floatX);
    }
}

// Allocate memory for the parameters and point the individual tensors to the right places.
inline void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes, size_t* param_sizeof){
    
    size_t num_parameters_bytes = 0;
    for(int i=0; i<NUM_PARAMETER_TENSORS; ++i){
        num_parameters_bytes += param_sizes[i] * param_sizeof[i];
    }

    // malloc all parameters at once on the device
    void* params_memory;
    cudaCheck(cudaMalloc((void**)&params_memory, num_parameters_bytes));
    floatX** ptrs[] = {
        &params->patch_embd_kernel, &params->patch_embd_bias, &params->cls_token, &params->pos_embd,
        &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb, &params->attn_projw, &params->attn_projb,
        &params->ln2w, &params->ln2b, &params->mlpw, &params->mlpb, &params->mlp_projw, &params->mlp_projb,
        &params->clsw, &params->clsb
    };
    char* params_memory_iterator = (char*)params_memory;
    for(int i=0; i<NUM_PARAMETER_TENSORS; ++i){
        *(ptrs[i]) = (floatX*)params_memory_iterator;
        params_memory_iterator += param_sizes[i] * param_sizeof[i];
    }
    return params_memory;
}

// ----------------------------------------------------------------------------
// ViT model activation tensors

constexpr const int NUM_ACTIVATION_TENSORS=21;
typedef struct{
    floatX* patch_embd; // (B, H, img_height/patch_size, img_width/path_size)
    floatX* encoded; // (batch_size (B), num_patches + 1 (T), hidden_size (H))
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    floatX* ln1; // layernorm1 output (num_layers (L), B, T, H)
    floatX* qkv; // matmul projection output (L, B, T, 3*H)
    floatX* preattn; // (L, B, num_heads (NH), T, T)
    floatX* attn; // (L, B, NH, T, T)
    floatX* attn_y; // attention output (L, B, T, H)
    floatX* attn_proj; // post attention projection output (L, B, T, H)
    floatX* resi_attn; // post attention residual output (L, B, T, H)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    floatX* ln2; // layernorm2 output (L, B, T, H)
    floatX* mlph; // MLP hidden layer output (L, B, T, 4*H)
    floatX* mlph_gelu; // gelu output (L, B, T, 4*H)
    floatX* mlp_proj; // MLP projection output (L, B, T, H)
    floatX* resi_mlp; // post mlp residual output (L, B, T, H)
    floatX* logits; // matmul output projection H to num_classes (B, 1, NC)
    floatX* probs; // softmax output (B, 1, NC);
    float* losses; // loss metric for optimization (B, 1, 1);
} ActivationTensors;

struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};

#define TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)};

inline void fill_in_activation_sizes(const ActivationTensors* data, TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS], size_t B, ViTConfig* config){

    size_t nL = config->num_layers;
    size_t nH = config->num_attention_heads;
    size_t H = config->hidden_size;
    size_t iW = config->image_width;
    size_t iH = config->image_height;
    size_t ps = config->patch_size;
    size_t nCls = config->num_classes;

    assert(iW%ps==0 && iH%ps==0); // Sanity check.
    size_t nP = (iW/ps)*(iH/ps); // number of patches.
    size_t T = nP + 1; // sequence length (+1 correspond to cls_token)

    tensors[0] = TENSOR_SPEC(data->patch_embd, B*H*nP);
    tensors[1] = TENSOR_SPEC(data->encoded, B*T*H);
    tensors[2] = TENSOR_SPEC(data->ln1_mean, nL*B*T);
    tensors[3] = TENSOR_SPEC(data->ln1_rstd, nL*B*T);
    tensors[4] = TENSOR_SPEC(data->ln1, nL*B*T*H);
    tensors[5] = TENSOR_SPEC(data->qkv, nL*B*T*3*H);
    tensors[6] = TENSOR_SPEC(data->preattn, nL*B*nH*T*T);
    tensors[7] = TENSOR_SPEC(data->attn, nL*B*nH*T*T);
    tensors[8] = TENSOR_SPEC(data->attn_y, nL*B*T*H);
    tensors[9] = TENSOR_SPEC(data->attn_proj, nL*B*T*H);
    tensors[10] = TENSOR_SPEC(data->resi_attn, nL*B*T*H);
    tensors[11] = TENSOR_SPEC(data->ln2_mean, nL*B*T);
    tensors[12] = TENSOR_SPEC(data->ln2_rstd, nL*B*T);
    tensors[13] = TENSOR_SPEC(data->ln2, nL*B*T*H);
    tensors[14] = TENSOR_SPEC(data->mlph, nL*B*T*4*H);
    tensors[15] = TENSOR_SPEC(data->mlph_gelu, nL*B*T*4*H);
    tensors[16] = TENSOR_SPEC(data->mlp_proj, nL*B*T*H);
    tensors[17] = TENSOR_SPEC(data->resi_mlp, nL*B*T*H);
    tensors[18] = TENSOR_SPEC(data->logits, B*nCls);
    tensors[19] = TENSOR_SPEC(data->probs, B*nCls);
    tensors[20] = TENSOR_SPEC(data->losses, B);
}

inline void* malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]){

    size_t num_bytes = 0;
    for(int i=0; i<NUM_ACTIVATION_TENSORS; ++i){
        num_bytes += tensors[i].size * sizeof_dtype(tensors[i].type);
    }

    printf("[INFO] Allocating %d MiB for activations.\n", (int)round(num_bytes/(1024*1024)));

    void* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, num_bytes));
    cudaCheck(cudaMemset(acts_memory, 0, num_bytes));

    char* acts_memory_iterator = (char*)acts_memory;
    for(size_t i=0; i<NUM_ACTIVATION_TENSORS; ++i){
        // extra protection so we don't accidentally use an empty buffer.
        if(tensors[i].size==0){
            *(tensors[i].ptr) = NULL;
        } else {
            *(tensors[i].ptr) = acts_memory_iterator;
            acts_memory_iterator += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }

    return acts_memory;
}

typedef struct{

    ViTConfig config;

    // The weights (params of the model), and their sizes.
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    size_t param_sizeof[NUM_PARAMETER_TENSORS];
    void* params_memory;
    size_t num_parameters;
    size_t num_parameters_bytes;

    // Gradients of the weights.
    ParameterTensors params_grads;
    void* params_grads_memory;

    // Buffers for the AdamW optimizer.
    float* m_memory;
    float* v_memory;
    
    // The activations of the model, and their sizes.
    ActivationTensors acts;
    TensorSpec acts_specs[NUM_ACTIVATION_TENSORS];
    void* acts_memory;

    // Gradients ofthe activations.
    ActivationTensors acts_grads;
    void* acts_grads_memory;

    // Other run state configurations (training data)
    int batch_size;
    int curr_batch_idx;
    float* inputs;
    int* targets;
    float mean_loss;
    int nImages;
    bool training_mode;

    // configurations (test data)
    float* inputs_test;
    int* targets_test;
    float mean_loss_test;
    int nImages_test;
    int batch_size_test;
    int curr_batch_idx_test;

    // Entirety of data
    float* data_train;
    float* data_test;
    int* labels_train;
    int* labels_test;

} ViTModel;