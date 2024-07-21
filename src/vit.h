#pragma once
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "conv2d.h"
#include "embeddings.h"
#include "layernorm.h"
#include "matmul.h"
#include "attention.h"
#include "residual.h"
#include "activations.h"
#include "softmax.h"

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

#define NUM_ACTIVATION_TENSORS 21
typedef struct{
    float* patch_embd; // (B, H, img_height/patch_size, img_width/path_size)
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
        &acts->patch_embd, &acts->encoded, &acts->ln1_mean, &acts->ln1_rstd, &acts->ln1, &acts->qkv,
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
    int channels; // 3
    int patch_size; // 4, therefore, for a 32 x 32 image, 8 x 8 patches are generated.
    int hidden_size; // 48
    int num_attention_heads; // 4, so head size = 48/4 = 12
    int num_layers; // L = 4
    int num_classes; // 10, CIFAR10 dataset
} ViTConfig;

typedef struct{
    ViTConfig config;

    // The weights (params) of the model, and their sizes.
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_params;
    // Gradients of the weights
    ParameterTensors params_grads;
    float* params_grads_memory;
    // Buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;

    // The activations of the model, and their sizes.
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // Gradients of the activations
    ActivationTensors acts_grads;
    float* acts_grads_memory;

    // Other run state configurations
    int batch_size;
    float* inputs;
    int* targets;
    float mean_loss;
    
} ViTModel;

inline void ViT_zero_grad(ViTModel* model){
    if(model->params_grads_memory != NULL){
        memset(model->params_grads_memory, 0, model->num_params*sizeof(float));
    }
    if(model->acts_grads_memory != NULL){
        memset(model->acts_grads_memory, 0, model->num_activations*sizeof(float));
    }
}

// ViT forward function
//
// @param model         Model config for the current ViT model.
// @param inputs        linearized input tensors (B, C, H, W)
// @param targets       linearized ground truth label tensors (B, 1, 1)
// @param B             number of batches
//
void ViT_forward(ViTModel* model, float* inputs, int* targets, int B);

// ViT backward function
//
// @param model         Model config for the current ViT model.
void ViT_backward(ViTModel* model);