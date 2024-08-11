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

constexpr const int NUM_PARAMETER_TENSORS=18;
typedef struct{
    float* patch_embd_kernel; // (hidden_size (H), num_channel(C), patch_height (PH), patch_width (PW))
    float* patch_embd_bias; // (H)
    float* cls_token; // (1, 1, H)
    float* pos_embd; // (1, 1 + num_patches (1+NP), H)
    float* ln1w; // (num_layers (L), H)
    float* ln1b; // (L, H)
    float* qkvw; // qkv projection matmul weight, (L, H, 3*H = 3*head_size*num_heads)
    float* qkvb; // (L, 3*H)
    float* attn_projw; // post attn matmul weight, (L, H, H)
    float* attn_projb; // (L, H)
    float* ln2w; // (L, H)
    float* ln2b; // (L, H)
    float* mlpw; // (L, H, 4*H)
    float* mlpb; // (L, 4*H)
    float* mlp_projw; // (L, 4*H, H)
    float* mlp_projb; // (L, H)
    float* clsw; // classifier matmul (H, num_classes (NC) )
    float* clsb; // (NC)
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

