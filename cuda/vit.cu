#include "vit.cuh"

void ViT_init_common(ViTModel* model){

    // Common inits outside of the model weights.
    // memory lazily initialized in forward() call.
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->inputs_test = NULL;
    model->targets_test = NULL;

    model->batch_size = 0;
    model->curr_batch_idx = 0;
    model->mean_loss = 0.f;
    model->nImages = 0;
    model->training_mode = true;

    model->batch_size_test = 0;
    model->curr_batch_idx_test = 0;
    model->mean_loss_test = 0.f;
    model->nImages_test = 0;

    model->data_train = NULL;
    model->data_test = NULL;
    model->labels_train = NULL;
    model->labels_test = NULL;

    // Get the device ID
    cudaGetDevice(&model->deviceId);

    // Get the device properties
    cudaGetDeviceProperties(&model->deviceProp, model->deviceId);

    // CUDA block size.
    model->max_num_threads = model->deviceProp.maxThreadsPerBlock;
    model->sqrt_max_num_threads = static_cast<int>(std::sqrt(model->max_num_threads));
    model->cubert_max_num_threads = static_cast<int>(std::cbrt(model->max_num_threads));

    printf("[INFO] Current CUDA deivce: %s, max threads/block = %d, sqrt max = %d, cubert max = %d\n", 
            model->deviceProp.name, model->max_num_threads, model->sqrt_max_num_threads, model->cubert_max_num_threads);

}

void ViT_allocate_weights(ViTModel* model){

    // fill in all the parameter tensor dimensions and types.
    fill_in_parameter_sizes(model->param_sizes, model->param_sizeof, &(model->config));
    model->num_parameters = 0;
    model->num_parameters_bytes = 0;
    for(int i=0; i<NUM_PARAMETER_TENSORS; ++i){
        model->num_parameters += model->param_sizes[i];
        model->num_parameters_bytes += model->param_sizes[i] * model->param_sizeof[i];
    }

    // Create memory for model parameters on the deivce.
    assert(model->params_memory == nullptr);
    model->params_memory = malloc_and_point_parameters(&(model->params), model->param_sizes, model->param_sizeof);
}

void ViT_allocate_states(ViTModel* model, int B){
    printf("[INFO] Allocating %d MiB for parameter gradients.\n", (int)round(model->num_parameters * sizeof(floatX)/(1024*1024)));
    assert(model->params_grads_memory == nullptr);
    model->params_grads_memory = malloc_and_point_parameters(&(model->params_grads), model->param_sizes, model->param_sizeof);

    model->batch_size = B;
    int im_C = model->config.channels;
    int im_H = model->config.image_height;
    int im_W = model->config.image_width;

    // Allocate the space for activation tensors and activation gradient tensors.
    fill_in_activation_sizes(&(model->acts), model->acts_specs, B, &(model->config));
    model->acts_memory = malloc_and_point_activations(model->acts_specs);
    fill_in_activation_sizes(&(model->acts_grads), model->acts_grads_specs, B, &(model->config));
    model->acts_grads_memory = malloc_and_point_activations(model->acts_grads_specs);
    size_t num_act_bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_act_bytes += model->acts_specs[i].size * sizeof_dtype(model->acts_specs[i].type);
    }
    num_act_bytes *= 2;
    printf("[INFO] Allocating %zd MiB for activation/gradients tensors.\n", num_act_bytes/(1024*1024));

    // Create memory for cahcing inputs and targets
    cudaCheck(cudaMalloc((void**)&model->inputs, B*im_C*im_H*im_W*sizeof(float)));
    cudaCheck(cudaMalloc((void**)&model->targets, B*sizeof(float)));
    cudaCheck(cudaMalloc((void**)&model->accumulated_mean_loss, sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&model->cpu_loss, sizeof(float)));

    // AdamW optimizer parameters.
    printf("[INFO] Allocating %zu MiB for AdamW optimizer state m.\n", sizeof(float)>>20);
    printf("[INFO] Allocating %zu MiB for AdamW optimizer state v.\n", sizeof(float)>>20);
    assert(model->m_memory == nullptr);
    assert(model->v_memory == nullptr);
    cudaCheck(cudaMalloc((void**)&model->m_memory, sizeof(float)));
    cudaCheck(cudaMalloc((void**)&model->v_memory, sizeof(float)));

    // Memory usage info
    size_t free, total;
    cudaCheck(cudaMemGetInfo(&free, &total));
    printf("[INFO] Device memory usage %zd MiB / %zd MiB.\n", (total-free)/(1024*1024), total/(1024*1024));
}

void ViT_forward(ViTModel* model, const float* inputs, const int* targets, size_t B){
    if(model->params_memory == NULL){
        printf("[ERROR] model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // Number parameters
    int NC = model->config.num_classes;
    int NL = model->config.num_layers;
    int NH = model->config.num_attention_heads;
    int im_C = model->config.channels;
    int im_H = model->config.image_height;
    int im_W = model->config.image_width;
    int H = model->config.hidden_size;
    int P = model->config.patch_size;

    // Sanity check
    assert(im_W%P==0 && im_H%P==0);

    int NP = (im_W/P)*(im_H/P); // number of patches 
    int T = NP + 1; // sequence length (+1 corresponds to cls_token)

    // Validate B is not larger than the values used at initialization.
    // Smaller B is okay for inference only.
    if(B > model->batch_size){
        printf("[ERROR] Model got B=%d, Desired: (max) B=%d.\n", (int)B, model->batch_size);
        exit(EXIT_FAILURE);
    }

    // Copy inputs, targets to the model.
    cudaCheck(cudaMemcpy(model->inputs, inputs, B*im_C*im_H*im_W*sizeof(float)));
    cudaCheck(cudaMemcpy(model->targets, targets, B*sizeof(int)));

    // Forward pass
    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;

    // Patch embedding
    conv2d_forward1(model->inputs, params.patch_embd_kernel, params.patch_embd_bias, acts.patch_embd,
                    B, im_C, im_H, im_W, H, P, P, P, 0, model->cubert_max_num_threads);
    
    // Embedding = pos_embedding + cat(cls_token, patch_embedding)
    embeddings_forward1(acts.patch_embd, params.cls_token, params.pos_embd, acts.encoded,
                        B, NP, H, im_H/P, im_W/P, model->cubert_max_num_threads);
    
    // Attention block layers
    floatX* residual;
    for(int l=0; l<NL; ++l){
        residual = l == 0 ? acts.encoded : acts.resi_mlp + (l-1)*B*T*H;

        // get the pointers of the weights for the current layer
        floatX* l_ln1w = params.ln1w + l*H;
        floatX* l_ln1b = params.ln1b + l*H;
        floatX* l_qkvw = params.qkvw + l*H*3*H; 
        floatX* l_qkvb = params.qkvb + l*3*H;
        floatX* l_attn_projw = params.attn_projw + l*H*H;
        floatX* l_attn_projb = params.attn_projb + l*H; 
        floatX* l_ln2w = params.ln2w + l*H; 
        floatX* l_ln2b = params.ln2b + l*H;
        floatX* l_mlpw = params.mlpw + l*H*4*H; 
        floatX* l_mlpb = params.mlpb + l*4*H; 
        floatX* l_mlp_projw = params.mlp_projw + l*4*H*H; 
        floatX* l_mlp_projb = params.mlp_projb + l*H;

        // get the pointers of the activations for the current layer
        float* l_ln1_mean = acts.ln1_mean + l*B*T; 
        float* l_ln1_rstd = acts.ln1_rstd + l*B*T; 
        floatX* l_ln1 = acts.ln1 + l*B*T*H; 
        floatX* l_qkv = acts.qkv + l*B*T*3*H; 
        floatX* l_preattn = acts.preattn + l*B*NH*T*T; 
        floatX* l_attn = acts.attn + l*B*NH*T*T; 
        floatX* l_attn_y = acts.attn_y + l*B*T*H; 
        floatX* l_attn_proj = acts.attn_proj + l*B*T*H; 
        floatX* l_resi_attn = acts.resi_attn + l*B*T*H; 
        float* l_ln2_mean = acts.ln2_mean + l*B*T;
        float* l_ln2_rstd = acts.ln2_rstd + l*B*T;
        floatX* l_ln2 = acts.ln2 + l*B*T*H; 
        floatX* l_mlph = acts.mlph + l*B*T*4*H;
        floatX* l_mlph_gelu = acts.mlph_gelu + l*B*T*4*H; 
        floatX* l_mlp_proj = acts.mlp_proj + l*B*T*H;
        floatX* l_resi_mlp = acts.resi_mlp + l*B*T*H;

        // attention block forward pass
        layernorm_forward1(residual, l_ln1_mean, l_ln1_rstd, l_ln1w, l_ln1b, l_ln1, B, T, H, model->max_num_threads);
        matmul_forward1(l_ln1, l_qkv, l_qkvw, l_qkvb, B, T, H, 3*H, model->sqrt_max_num_threads);
        attention_forward1(l_qkv, l_preattn, l_attn, l_attn_y, B, T, H, NH, model->max_num_threads);
        matmul_forward1(l_attn_y, l_attn_proj, l_attn_projw, l_attn_projb, B, T, H, H, model->sqrt_max_num_threads);
        residual_forward1(l_attn_proj, residual, l_resi_attn, B*T*H, model->max_num_threads);
        layernorm_forward1(l_resi_attn, l_ln2_mean, l_ln2_rstd, l_ln2w, l_ln2b, l_ln2, B, T, H, model->max_num_threads);
        matmul_forward1(l_ln2, l_mlph, l_mlpw, l_mlpb, B, T, H, 4*H, model->sqrt_max_num_threads);
        gelu_forward1(l_mlph, l_mlph_gelu, B*T*4*H, model->max_num_threads);
        matmul_forward1(l_mlph_gelu, l_mlp_proj, l_mlp_projw, l_mlp_projb, B, T, 4*H, H, model->sqrt_max_num_threads);
        residual_forward1(l_mlp_proj, l_resi_attn, l_resi_mlp, B*T*H, model->max_num_threads);
    }
    residual = acts.resi_mlp + (NL-1)*B*T*H; // (B, T, H)

    // classifier
    // The first index in the sequence T, corresponding to cls_token is responsible for 
    // classification prediction.
    matmul_forward_with_slicing_at_t2(residual, acts.logits, params.clsw, params.clsb, B, T, H, NC, 0, model->sqrt_max_num_threads);
    softmax_forward1(acts.logits, acts.probs, B, NC, model->max_num_threads);
    crossentropy_forward1(acts.probs, targets, acts.losses, B, NC, model->max_num_threads);

    // loss metric calculation for the model.
    // calculating mean loss on device.
    *(model->accumulated_mean_loss) = 0.f;
    for(int b=0; b<B; ++b){
        *(model->accumulated_mean_loss) += acts.losses[b];
    }

    // mean loss to be copied to host at logging step (completion of forward/backward/update cycle for the total steps)

}

void ViT_backward(ViTModel* model){
    if(model->params_grads_memory == nullptr || model->acts_grads_memory == nullptr){
        fprintf(stderr, "[ERROR] Need to allocate gradients before backward pass call.\n");
        exit(EXIT_FAILURE);
    }

    // Number parameters
    int B = model->batch_size;
    int NC = model->config.num_classes;
    int NL = model->config.num_layers;
    int NH = model->config.num_attention_heads;
    int im_C = model->config.channels;
    int im_H = model->config.image_height;
    int im_W = model->config.image_width;
    int H = model->config.hidden_size;
    int P = model->config.patch_size;
    int NP = (im_W/P)*(im_H/P);
    int T = NP + 1;

    // Sanity check was performed during forward pass, so safe to skip

    // Backward pass
    ParameterTensors params = model->params;
    ParameterTensors param_grads = model->params_grads;
    ActivationTensors acts = model->acts;
    ActivationTensors acts_grads = model->acts_grads;

    // Start the chainrule by filling in dlosses with 1.f/B
    float dloss_mean = 1.f/B;
    for(int b=0; b<B; ++b) {acts_grads.losses[b] = dloss_mean;}

    crossentropy_softmax_backward1(acts.probs, model->targets, acts_grads.logits, acts_grads.losses, B, NC, model->max_num_threads);
    floatX* residual = acts.resi_mlp + (NL-1)*B*T*H; // (B, T, H)
    floatX* dresidual = acts_grads.resi_mlp + (NL-1)*B*T*H;

}