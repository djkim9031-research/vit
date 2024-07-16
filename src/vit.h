#pragma once

typedef struct{
    float* patch_embd_kernal; // (hidden_size (H), num_channel(C), patch_height (PH), patch_width (PW))
    float* patch_embd_bias; // (H)
    float* cls_token; // (1, 1, H)
    float* pos_embd; // (1, 1 + num_patches (1+NP), H)
    float* ln1w; // (num_layers (L), H)
    float* ln1b; // (L, H)
    float* qkvw; // qkv projection matmul weight, (L, H, 3*H = 3*head_size*num_heads)
    float* qkvb; // (L, 3*H)
    float* attnprojw; // post attn matmul weight, (L, H, H)
    float* attnprojb; // (L, H)
    float* ln2w; // (L, H)
    float* ln2b; // (L, H)
    float* fcw; // (L, H, 4*H)
    float* fcb; // (L, 4*H)
    float* fcprojw; // (L, 4*H, H)
    float* fcprojb; // (L, H)
    float* clsw; // classifier matmul (H, num_classes (NC) )
    float* clsb; // (NC)
} ParameterTensors;