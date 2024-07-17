#pragma once
#include <stddef.h>
#include <stdlib.h>

#define NUM_PARAMETER_TENSORS 18
typedef struct{
    float* patch_embd_kernal; // (hidden_size (H), num_channel(C), patch_height (PH), patch_width (PW))
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

// Allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes){
    size_t num_parameters = 0;
    for(size_t i=0; i<NUM_PARAMETER_TENSORS; ++i){
        num_parameters += param_sizes[i];
    }
    
    // malloc all parameters at once
    float* params_memory = (float*)malloc(num_parameters * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &params->patch_embd_kernal, &params->patch_embd_bias, &params->cls_token, &params->pos_embd,
        &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb, &params->attn_projw, &params->attn_projb,
        &params->ln2w, &params->ln2b, &params->mlpw, &params->mlpb, &params->mlp_projw, &params->mlp_projb,
        &params->clsw, &params->clsb
    };

    float* params_memory_iterators = params_memory;
    for(size_t i=0; i<NUM_PARAMETER_TENSORS; ++i){
        *(ptrs[i]) = params_memory_iterators;
        params_memory_iterators += param_sizes[i];
    }
    
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 20
typedef struct{
    float* encoded; // (batch_size (B), num_patches + 1 (T), hidden_size (H))
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* ln1; // layernorm1 output (num_layers (L), B, T, H)
    float* qkv; // matmul projection output (L, B, T, 3*H)
    float* preattn; // (L, B, num_heads (NH), T, T)
    float* attn; // (L, B, NH, T, T)
    float* attn_y; // attention output (L, B, T, H)
    float* attn_proj; // post attention projection output (L, B, T, H)
    float* resi_attn; // post attention residual output (L, B, T, H)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* ln2; // layernorm2 output (L, B, T, H)
    float* mlph; // MLP hidden layer output (L, B, T, 4*H)
    float* mlph_gelu; // gelu output (L, B, T, 4*H)
    float* mlp_proj; // MLP projection output (L, B, T, H)
    float* resi_mlp; // post mlp residual output (L, B, T, H)
    float* logits; // matmul output projection H to num_classes (B, 1, NC)
    float* probs; // softmax output (B, 1, NC);
    float* losses; // loss metric for optimization (B, 1, 1);
} ActivationTensors;

// Allocate memory for the activation tensors and point the individual tensors to the right places
float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes){
    size_t num_activations = 0;
    for(size_t i=0; i<NUM_ACTIVATION_TENSORS; ++i){
        num_activations += act_sizes[i];
    }
    
    // malloc all parameters
    float* acts_memory = (float*)malloc(num_activations * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &acts->encoded, &acts->ln1_mean, &acts->ln1_rstd, &acts->ln1, &acts->qkv,
        &acts->preattn, &acts->attn, &acts->attn_y, &acts->attn_proj, &acts->resi_attn,
        &acts->ln2_mean, &acts->ln2_rstd, &acts->ln2, &acts->mlph, &acts->mlph_gelu,
        &acts->mlp_proj, &acts->resi_mlp, &acts->logits, &acts->probs, &acts->losses
    };

    float* acts_memory_iterators = acts_memory;
    for(size_t i=0; i<NUM_ACTIVATION_TENSORS; ++i){
        *(ptrs[i]) = acts_memory_iterators;
        acts_memory_iterators += act_sizes[i];
    }
    
    return acts_memory;
}

typedef struct{
    int image_width; // 32
    int image_height; // 32
    int num_channels; // 3
    int patch_size; // 4, therefore, for a 32 x 32 image, 8 x 8 patches are generated.
    int hidden_size; // 48
    int num_attention_heads; // 4, so head size = 48/4 = 12
    int num_layers; // L = 4
    int num_classes; // 10, CIFAR10 dataset
} ViTConfig;