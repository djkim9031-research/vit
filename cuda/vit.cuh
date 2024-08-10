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
