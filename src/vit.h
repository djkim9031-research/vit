#pragma once
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/stat.h>

#include "utils.h"
#include "conv2d.h"
#include "embeddings.h"
#include "layernorm.h"
#include "matmul.h"
#include "attention.h"
#include "residual.h"
#include "activations.h"
#include "softmax.h"
#include "dataprep.h"
#include "init_weights.h"

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
inline float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes){
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
inline float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes){
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

// Zero-ing out all the gradients.
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
// @param model                 Model config for the current ViT model.
// @param inputs                linearized input tensors (B, C, H, W)
// @param targets               linearized ground truth label tensors (B, 1, 1)
// @param B                     number of batches
//
void ViT_forward(ViTModel* model, float* inputs, int* targets, int B);

// ViT backward function
//
// @param model                 Model config for the current ViT model.
void ViT_backward(ViTModel* model);

// ViT update function. This is called once the forward-backward pass are made, and works as an AdamW optimizer.
// reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
//
// @param model                 Model config for the current ViT model.
// @param learning_rate         Learning rate of the optimizer
// @param beta1                 First coefficient to compute running averages of gradients and its square
// @param beta2                 Second coefficient to compute running averages of gradients and its square
// @param eps                   Term added to the denominator to improve numerical stability
// @param weight_decay          Weight decay coefficient
// @param t                     Current gradient step
void ViT_update(ViTModel* model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t);

// Freeing the allocated memories.
inline void ViT_free(ViTModel* model){
    free(model->params_memory);
    free(model->params_grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->acts_grads_memory);
    free(model->inputs);
    free(model->targets);
    free(model->inputs_test);
    free(model->targets_test);
    free(model->data_train);
    free(model->data_test);
    free(model->labels_train);
    free(model->labels_test);
}

// Dataloader function.
// It will read the image dataset as linearized 1D array and labels.
// Then preprocess the read data accordingly, 
// and chunk them to linearized batches.
// 
// @param model                 Model config for the current ViT model. 
// @param data_dir              Directory where train/test dataset and labels are stored.
//                              .bmp images and label.txt files should exist under `data_dir/train`
//                              and `data_dir/test` folders.
// @param nData_to_read_train   number of train dataset to read.
// @param nData_to_read_test    number of test dataset to read.
//
void Dataloader(ViTModel* model, const char* data_dir, int nData_to_read_train, int nData_to_read_test);

// Function to get the batch of data in sequential order.
// In the `dataloader`, all the pixel data and labels are extracted and shuffled.
// Therefore, in this function call, batch data are extracted sequentially from the entirety of the dataset.
// The sequential batch is tracked with `curr_batch_idx` inside the ViTModel struct.
//
// @param model                 Model config for the current ViT model. 
// @param batch_data            Linearized batch pixel data input (batch_size, channel, height, width)
// @param batch_labels          Linearized batch target (batch_size, cls_idx)
// @param batch_size            number of data being handled in current batch.
// 
void GetBatch(ViTModel* model, float* batch_data, int* batch_labels, int& batch_size);

// Build the ViT model from YAML file.
//
// @param model                 Model config for the current ViT model. 
// @param yaml_path             Path to the YAML file.
void ViT_from_YAML(ViTModel* model, const char* yaml_path);

// Initialize ViT model after the YAML read call.
//
// @param model                 Model config for the current ViT model. 
//
void ViT_init(ViTModel* model);

// Initialize ViT model trainable parameters.
// 
// @param parameters            The trainable parameter set
// @param param_sizes           Size_t array containing the size of each trainable parameter
//
inline void param_initializer(ParameterTensors* parameters, size_t* param_sizes){

    float normal_mean = 0.f;
    float normal_std = 0.02f;
    float rand_seed = 42;
    
    // Conv2d and matmul operation weights are initialized from 
    // normal distribution with 0 mean and 0.02 std.
    normal_init(parameters->patch_embd_kernal, param_sizes[0], normal_mean, normal_std, rand_seed);
    normal_init(parameters->qkvw, param_sizes[6], normal_mean, normal_std, rand_seed);
    normal_init(parameters->attn_projw, param_sizes[8], normal_mean, normal_std, rand_seed);
    normal_init(parameters->mlpw, param_sizes[12], normal_mean, normal_std, rand_seed);
    normal_init(parameters->mlp_projw, param_sizes[14], normal_mean, normal_std, rand_seed);
    normal_init(parameters->clsw, param_sizes[16], normal_mean, normal_std, rand_seed);

    // Conv2d and matmul operation biases are initialized to 0
    zeros_init(parameters->patch_embd_bias, param_sizes[1]); 
    zeros_init(parameters->qkvb, param_sizes[7]); 
    zeros_init(parameters->attn_projb, param_sizes[9]);
    zeros_init(parameters->mlpb, param_sizes[13]); 
    zeros_init(parameters->mlp_projb, param_sizes[15]); 
    zeros_init(parameters->clsb, param_sizes[17]); 

    // Layernorm operation weights are initialized to 1
    ones_init(parameters->ln1w, param_sizes[4]);
    ones_init(parameters->ln2w, param_sizes[10]);
    
    // Layernorm operation biases are initialized to 0
    zeros_init(parameters->ln1b, param_sizes[5]);
    zeros_init(parameters->ln2b, param_sizes[11]);

    // Class token and position embedding weights are initialized from
    // normal distribution with 0 mean and 0.02 std with [-2, 2] truncation
    trunc_normal_init(parameters->cls_token, param_sizes[2], normal_mean, normal_std, -2.0f, 2.0f, rand_seed);
    trunc_normal_init(parameters->pos_embd, param_sizes[3], normal_mean, normal_std, -2.0f, 2.0f, rand_seed);
}

// ViT model training function call.
//
// @param yaml_path             Path to the YAML file.
// @param data_dir              Directory where train/test dataset and labels are stored.
//                              .bmp images and label.txt files should exist under `data_dir/train`
//                              and `data_dir/test` folders.
// @param param_file_dir        Path to the directory where binary file exists or should be saved under (if not existing yet).
// @param filename              Binary file name.
// @param nData_to_read_train   number of train dataset to read.
// @param nData_to_read_test    number of test dataset to read.
//
void ViT_trainer(const char* yaml_path, const char* data_dir, const char* param_file_dir, const char* filename, int nData_to_read_train, int nData_to_read_test);

// ViT model evaluation functon call.
//
// @param model                 Model config for the current ViT model. 
void ViT_evaluate(ViTModel* model);

// Function to save weight/bias tensors to a binary file.
//
// @param params                Parameter tensors to save.
// @param file_path             Path to the output binary file.
// @param param_size            Array of sizes of each parameter tensor.
//
inline void save_parameters(const ParameterTensors* params, const char* file_path, size_t* param_sizes){
    
    FILE* file = fopen(file_path, "wb");
    if(!file){
        fprintf(stderr, "Failed to open file for saving parameter tensors.\n");
        return;
    }

    float* ptrs[] = {
        params->patch_embd_kernal, params->patch_embd_bias, params->cls_token, params->pos_embd,
        params->ln1w, params->ln1b, params->qkvw, params->qkvb, params->attn_projw, params->attn_projb,
        params->ln2w, params->ln2b, params->mlpw, params->mlpb, params->mlp_projw, params->mlp_projb,
        params->clsw, params->clsb
    };

    for(size_t i=0; i<NUM_PARAMETER_TENSORS; ++i){
        fwrite(ptrs[i], sizeof(float), param_sizes[i], file);
    }

    fclose(file);
}

// Function to load weight/bias tensors from a binary file.
//
// @param params                Parameter tensors to load.
// @param file_path             Path to the input binary file.
// @param param_size            Array of sizes of each parameter tensor.
//
inline void load_parameters(ParameterTensors* params, const char* file_path, size_t* param_sizes){

    FILE* file = fopen(file_path, "rb");
    if(!file){
        fprintf(stderr, "No saved parameters found.\n");
        return;
    }

    float* ptrs[] = {
        params->patch_embd_kernal, params->patch_embd_bias, params->cls_token, params->pos_embd,
        params->ln1w, params->ln1b, params->qkvw, params->qkvb, params->attn_projw, params->attn_projb,
        params->ln2w, params->ln2b, params->mlpw, params->mlpb, params->mlp_projw, params->mlp_projb,
        params->clsw, params->clsb
    };

    for(size_t i=0; i<NUM_PARAMETER_TENSORS; ++i){
        fread(ptrs[i], sizeof(float), param_sizes[i], file);
    }

    fclose(file);

    printf("------------------------------------------------------------------------\n");
    printf("Loaded the learnable parameter tensors from: %s\n", file_path);
}

// Helper function to check if the file exists.
//
// @param file_path             Path to the input binary file.
//
inline int file_exists(const char* file_path){
    struct stat buffer;
    return (stat(file_path, &buffer) == 0);
}

// Helper function to concatenate two strings, where one of which should first be converted from int to string.
//
// @param file_path_base        (const char*) file path base.
// @param epoch                 (int) current epoch.
// @param result_path           Concatenated file path.
//
inline void concatenate_file_path(const char* file_path_base, int epoch, char* result_path){

    // Convert the integer to a string.
    char i[20];
    snprintf(i, sizeof(i), "%d", epoch);

    // Concate the strings.
    strcpy(result_path, file_path_base);
    strcat(result_path, i);

    return;
}


