#include "vit.h"

void ViT_forward(ViTModel* model, float* inputs, int* targets, int B){
    // Ensure the model was initialized 
    if(model->params_memory == NULL){
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // Number parameters
    int NC = model->config.num_classes;
    int NL = model->config.num_layers;
    int NH = model->config.num_attention_heads;
    int C = model->config.channels;
    int H = model->config.image_height;
    int W = model->config.image_width;
    int H = model->config.hidden_size;
    int P = model->config.patch_size;

    // Sanity check
    assert(W%P==0 && H&P==0);

    int NP = (W/P)*(H/P); // number of patches 
    int T = NP + 1; // sequence length (+1 corresponds to cls_token)

    // Allocate space for all the activation tensors.
    if(model->acts_memory == NULL){
        model->batch_size = B;

        model->act_sizes[0] = B*T*H; // encoded
        model->act_sizes[1] = NL*B*T; // ln1_mean
        model->act_sizes[2] = NL*B*T; // ln1_rstd
        model->act_sizes[3] = NL*B*T*H; // ln1
        model->act_sizes[4] = NL*B*T*3*H; // qkv
        model->act_sizes[5] = NL*B*NH*T*T; // preattn
        model->act_sizes[6] = NL*B*NH*T*T; // attn
        model->act_sizes[7] = NL*B*T*H; // attn_y
        model->act_sizes[8] = NL*B*T*H; // attn_proj
        model->act_sizes[9] = NL*B*T*H; // resi_attn
        model->act_sizes[10] = NL*B*T; // ln2_mean
        model->act_sizes[11] = NL*B*T; // ln2_rstd
        model->act_sizes[12] = NL*B*T*H; // ln2
        model->act_sizes[13] = NL*B*T*4*H; // mlph
        model->act_sizes[14] = NL*B*T*4*H; // mlph_gelu
        model->act_sizes[15] = NL*B*T*H; // mlp_proj
        model->act_sizes[16] = NL*B*T*H; // mlp_resi
        model->act_sizes[17] = B*NC; // logits
        model->act_sizes[18] = B*NC; // probs
        model->act_sizes[19] = B; // losses

        size_t num_activations = 0;
        for(size_t i=0; i<NUM_ACTIVATION_TENSORS; ++i){
            num_activations += model->act_sizes[i];
        }
        printf("num_activation_tensors: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);

        // Create memory for caching inputs and targets
        model->inputs = (float*)malloc(B*C*H*W*sizeof(float));
        model->targets = (int*)malloc(B*sizeof(int));
    } 

    // Cache the inputs/targets in model struct.
    memcpy(model->inputs, inputs, B*C*H*W*sizeof(float));
    memcpy(model->targets, targets, B*sizeof(int));
}