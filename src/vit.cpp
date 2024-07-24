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
    int im_C = model->config.channels;
    int im_H = model->config.image_height;
    int im_W = model->config.image_width;
    int H = model->config.hidden_size;
    int P = model->config.patch_size;

    // Sanity check
    assert(im_W%P==0 && im_H&P==0);

    int NP = (im_W/P)*(im_H/P); // number of patches 
    int T = NP + 1; // sequence length (+1 corresponds to cls_token)

    // Allocate space for all the activation tensors.
    if(model->acts_memory == NULL){
        model->batch_size = B;

        model->act_sizes[0] = B*H*NP; // patch_embd
        model->act_sizes[1] = B*T*H; // encoded
        model->act_sizes[2] = NL*B*T; // ln1_mean
        model->act_sizes[3] = NL*B*T; // ln1_rstd
        model->act_sizes[4] = NL*B*T*H; // ln1
        model->act_sizes[5] = NL*B*T*3*H; // qkv
        model->act_sizes[6] = NL*B*NH*T*T; // preattn
        model->act_sizes[7] = NL*B*NH*T*T; // attn
        model->act_sizes[8] = NL*B*T*H; // attn_y
        model->act_sizes[9] = NL*B*T*H; // attn_proj
        model->act_sizes[10] = NL*B*T*H; // resi_attn
        model->act_sizes[11] = NL*B*T; // ln2_mean
        model->act_sizes[12] = NL*B*T; // ln2_rstd
        model->act_sizes[13] = NL*B*T*H; // ln2
        model->act_sizes[14] = NL*B*T*4*H; // mlph
        model->act_sizes[15] = NL*B*T*4*H; // mlph_gelu
        model->act_sizes[16] = NL*B*T*H; // mlp_proj
        model->act_sizes[17] = NL*B*T*H; // mlp_resi
        model->act_sizes[18] = B*NC; // logits
        model->act_sizes[19] = B*NC; // probs
        model->act_sizes[20] = B; // losses

        size_t num_activations = 0;
        for(size_t i=0; i<NUM_ACTIVATION_TENSORS; ++i){
            num_activations += model->act_sizes[i];
        }
        printf("num_activation_tensors: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);

        // Create memory for caching inputs and targets
        model->inputs = (float*)malloc(B*im_C*im_H*im_W*sizeof(float));
        model->targets = (int*)malloc(B*sizeof(int));
    } 

    // Cache the inputs/targets in model struct.
    memcpy(model->inputs, inputs, B*im_C*im_H*im_W*sizeof(float));
    memcpy(model->targets, targets, B*sizeof(int));

    // Forward pass
    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    
    // Patch embedding
    conv2d_forward(inputs, params.patch_embd_kernal, params.patch_embd_bias, acts.patch_embd,
                   B, im_C, im_H, im_W, H, P, P, P);
    // Embedding = pos_embedding + cat(cls_token, patch_embedding)
    embeddings_forward(acts.patch_embd, params.cls_token, params.pos_embd, acts.encoded,
                       B, NP, H, im_H/P, im_W/P);
    
    // Attention block layers
    float* residual;
    for(int l=0; l<NL; ++l){
        residual = l == 0 ? acts.encoded : acts.resi_mlp + (l-1)*B*T*H;

        // get the pointers of the weights for the current layer
        float* l_ln1w = params.ln1w + l*H;
        float* l_ln1b = params.ln1b + l*H;
        float* l_qkvw = params.qkvw + l*H*3*H; 
        float* l_qkvb = params.qkvb + l*3*H;
        float* l_attn_projw = params.attn_projw + l*H*H;
        float* l_attn_projb = params.attn_projb + l*H; 
        float* l_ln2w = params.ln2w + l*H; 
        float* l_ln2b = params.ln2b + l*H;
        float* l_mlpw = params.mlpw + l*H*4*H; 
        float* l_mlpb = params.mlpb + l*4*H; 
        float* l_mlp_projw = params.mlp_projw + l*4*H*H; 
        float* l_mlp_projb = params.mlp_projb + l*H;

        // get the pointers of the activations for the current layer
        float* l_ln1_mean = acts.ln1_mean + l*B*T; 
        float* l_ln1_rstd = acts.ln1_rstd + l*B*T; 
        float* l_ln1 = acts.ln1 + l*B*T*H; 
        float* l_qkv = acts.qkv + l*B*T*3*H; 
        float* l_preattn = acts.preattn + l*B*NH*T*T; 
        float* l_attn = acts.attn + l*B*NH*T*T; 
        float* l_attn_y = acts.attn_y + l*B*T*H; 
        float* l_attn_proj = acts.attn_proj + l*B*T*H; 
        float* l_resi_attn = acts.resi_attn + l*B*T*H; 
        float* l_ln2_mean = acts.ln2_mean + l*B*T;
        float* l_ln2_rstd = acts.ln2_rstd + l*B*T;
        float* l_ln2 = acts.ln2 + l*B*T*H; 
        float* l_mlph = acts.mlph + l*B*T*4*H;
        float* l_mlph_gelu = acts.mlph_gelu + l*B*T*4*H; 
        float* l_mlp_proj = acts.mlp_proj + l*B*T*H;
        float* l_resi_mlp = acts.resi_mlp + l*B*T*H;

        // attention block forward pass
        layernorm_forward(residual, l_ln1_mean, l_ln1_rstd, l_ln1w, l_ln1b, l_ln1, B, T, H);
        matmul_forward(l_ln1, l_qkv, l_qkvw, l_qkvb, B, T, H, 3*H);
        attention_forward(l_qkv, l_preattn, l_attn, l_attn_y, B, T, H, NH);
        matmul_forward(l_attn_y, l_attn_proj, l_attn_projw, l_attn_projb, B, T, H, H);
        residual_forward(l_attn_proj, residual, l_resi_attn, B*T*H);
        layernorm_forward(l_resi_attn, l_ln2_mean, l_ln2_rstd, l_ln2w, l_ln2b, l_ln2, B, T, H);
        matmul_forward(l_ln2, l_mlph, l_mlpw, l_mlpb, B, T, H, 4*H);
        gelu_forward(l_mlph, l_mlph_gelu, B*T*4*H);
        matmul_forward(l_mlph_gelu, l_mlp_proj, l_mlp_projw, l_mlp_projb, B, T, 4*H, H);
        residual_forward(l_mlp_proj, l_resi_attn, l_resi_mlp, B*T*H);
    }
    residual = acts.resi_mlp + (NL-1)*B*T*H; // (B, T, H)

    // classifier
    // The first index in the sequence T, corresponding to cls_token is responsible for 
    // classification prediction.
    matmul_forward_with_slicing_at_t(residual, acts.logits, params.clsw, params.clsb, B, T, H, NC, 0);
    softmax_forward(acts.logits, acts.probs, B, NC);
    crossentropy_forward(acts.probs, targets, acts.losses, B, NC);

    // loss metric calculation for the model
    float mean_loss = 0.f;
    for(int b=0; b<B; ++b){
        mean_loss += acts.losses[b];
    }
    mean_loss /= B;
    model->mean_loss = mean_loss;
}

void ViT_backward(ViTModel* model){

    // Allocate the memory for gradients of weights and activations if memory==NULL
    if(model->params_grads_memory == NULL){
        model->params_grads_memory = malloc_and_point_parameters(&model->params_grads, model->param_sizes);
        model->acts_grads_memory = malloc_and_point_activations(&model->acts_grads, model->act_sizes);
        ViT_zero_grad(model);
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

    crossentropy_softmax_backward(acts.probs, model->targets, acts_grads.logits, acts_grads.losses, B, NC);
    float* residual = acts.resi_mlp + (NL-1)*B*T*H; // (B, T, H)
    float* dresidual = acts_grads.resi_mlp + (NL-1)*B*T*H;
    matmul_backward_with_slicing_at_t(residual, params.clsw, dresidual, param_grads.clsw, param_grads.clsb,
                                      acts_grads.logits, B, T, H, NC, 0);
    for(int l=NL-1; l>=0; --l){
        residual = l == 0 ? acts.encoded : acts.resi_mlp + (l-1)*B*T*H;
        dresidual = l == 0 ? acts_grads.encoded : acts_grads.resi_mlp + (l-1)*B*T*H;

        // get the pointers of the weights for the current layer
        float* l_ln1w = params.ln1w + l*H;
        float* l_qkvw = params.qkvw + l*H*3*H; 
        float* l_attn_projw = params.attn_projw + l*H*H;
        float* l_ln2w = params.ln2w + l*H; 
        float* l_mlpw = params.mlpw + l*H*4*H; 
        float* l_mlp_projw = params.mlp_projw + l*4*H*H; 
        // get the pointers of the gradients of weights for the current layer
        float* dl_ln1w = param_grads.ln1w + l*H;
        float* dl_ln1b = param_grads.ln1b + l*H;
        float* dl_qkvw = param_grads.qkvw + l*H*3*H; 
        float* dl_qkvb = param_grads.qkvb + l*3*H;
        float* dl_attn_projw = param_grads.attn_projw + l*H*H;
        float* dl_attn_projb = param_grads.attn_projb + l*H; 
        float* dl_ln2w = param_grads.ln2w + l*H; 
        float* dl_ln2b = param_grads.ln2b + l*H;
        float* dl_mlpw = param_grads.mlpw + l*H*4*H; 
        float* dl_mlpb = param_grads.mlpb + l*4*H; 
        float* dl_mlp_projw = param_grads.mlp_projw + l*4*H*H; 
        float* dl_mlp_projb = param_grads.mlp_projb + l*H;

        // get the pointers of the activations for the current layer
        float* l_ln1_mean = acts.ln1_mean + l*B*T; 
        float* l_ln1_rstd = acts.ln1_rstd + l*B*T; 
        float* l_ln1 = acts.ln1 + l*B*T*H; 
        float* l_qkv = acts.qkv + l*B*T*3*H; 
        float* l_attn = acts.attn + l*B*NH*T*T; 
        float* l_attn_y = acts.attn_y + l*B*T*H; 
        float* l_resi_attn = acts.resi_attn + l*B*T*H; 
        float* l_ln2_mean = acts.ln2_mean + l*B*T;
        float* l_ln2_rstd = acts.ln2_rstd + l*B*T;
        float* l_ln2 = acts.ln2 + l*B*T*H; 
        float* l_mlph = acts.mlph + l*B*T*4*H;
        float* l_mlph_gelu = acts.mlph_gelu + l*B*T*4*H; 
        // get the pointers of the gradients of activations for the current layer
        float* dl_ln1 = acts_grads.ln1 + l*B*T*H; 
        float* dl_qkv = acts_grads.qkv + l*B*T*3*H; 
        float* dl_preattn = acts_grads.preattn + l*B*NH*T*T; 
        float* dl_attn = acts_grads.attn + l*B*NH*T*T; 
        float* dl_attn_y = acts_grads.attn_y + l*B*T*H; 
        float* dl_attn_proj = acts_grads.attn_proj + l*B*T*H; 
        float* dl_resi_attn = acts_grads.resi_attn + l*B*T*H; 
        float* dl_ln2 = acts_grads.ln2 + l*B*T*H; 
        float* dl_mlph = acts_grads.mlph + l*B*T*4*H;
        float* dl_mlph_gelu = acts_grads.mlph_gelu + l*B*T*4*H; 
        float* dl_mlp_proj = acts_grads.mlp_proj + l*B*T*H;
        float* dl_resi_mlp = acts_grads.resi_mlp + l*B*T*H;

        residual_backward(dl_mlp_proj, dl_resi_attn, dl_resi_mlp, B*T*H);
        matmul_backward(l_mlph_gelu, l_mlp_projw, dl_mlph_gelu, dl_mlp_projw, dl_mlp_projb, dl_mlp_proj, B, T, 4*H, H);
        gelu_backward(l_mlph, dl_mlph, dl_mlph_gelu, B*T*4*H);
        matmul_backward(l_ln2, l_mlpw, dl_ln2, dl_mlpw, dl_mlpb, dl_mlph, B, T, H, 4*H);
        layernorm_backward(l_resi_attn, l_ln2_mean, l_ln2_rstd, l_ln2w, dl_resi_attn, dl_ln2w, dl_ln2b, dl_ln2, B, T, H);
        residual_backward(dl_attn_proj, dresidual, dl_resi_attn, B*T*H);
        matmul_backward(l_attn_y, l_attn_projw, dl_attn_y, dl_attn_projw, dl_attn_projb, dl_attn_proj, B, T, H, H);
        attention_backward(l_qkv, l_attn, dl_qkv, dl_preattn, dl_attn, dl_attn_y, B, T, H, NH);
        matmul_backward(l_ln1, l_qkvw, dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, B, T, H, 3*H);
        layernorm_backward(residual, l_ln1_mean, l_ln1_rstd, l_ln1w, dresidual, dl_ln1w, dl_ln1b, dl_ln1, B, T, H);
    }
    embeddings_backward(acts.patch_embd, acts_grads.patch_embd, param_grads.cls_token, param_grads.pos_embd, acts_grads.encoded, B, NP, H, im_H/P, im_W/P);
    conv2d_backward(model->inputs, params.patch_embd_kernal, NULL, param_grads.patch_embd_kernal, param_grads.patch_embd_bias, acts_grads.patch_embd, B, im_C, im_H, im_W, H, P, P, P);
}

void ViT_update(ViTModel* model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t){
    // Allocate the memory for m_memory and v_memory if memory is NULL
    if(model->m_memory == NULL){
        model->m_memory = (float*)calloc(model->num_params, sizeof(float));
        model->v_memory = (float*)calloc(model->num_params, sizeof(float));
    }

    // Gradient updates
    for(int i=0; i<model->num_params; ++i){
        float param = model->params_memory[i];
        float grad = model->params_grads_memory[i];

        // Update the first moment (momentum)
        float m = beta1 * model->m_memory[i] + (1.f - beta1)*grad;
        // Update the second moment (RMSprop)
        float v = beta2 * model->v_memory[i] + (1.f - beta2)*grad*grad;
        // Bias corrections
        float m_hat = m / (1.f - powf(beta1, t));
        float v_hat = v / (1.f - powf(beta2, t));

        // Update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}


void Dataloader(ViTModel* model, const char* data_dir){

    const char* train_folder_name = "train/";
    const char* test_folder_name = "test/";
    const char* label_name = "labels.txt";

    size_t train_dir_len = strlen(data_dir) + strlen(train_folder_name) + 1;
    size_t test_dir_len = strlen(data_dir) + strlen(test_folder_name) + 1;
    size_t train_label_path_len = strlen(data_dir) + strlen(train_folder_name) + strlen(label_name) + 1;
    size_t test_label_path_len = strlen(data_dir) + strlen(test_folder_name) + strlen(label_name) + 1;

    char* train_path = (char*)malloc(train_dir_len*sizeof(char));
    char* test_path = (char*)malloc(test_dir_len*sizeof(char));
    char* train_label_path = (char*)malloc(train_label_path_len*sizeof(char));
    char* test_label_path = (char*)malloc(test_label_path_len*sizeof(char));

    if(train_path == NULL || test_path == NULL || 
       train_label_path == NULL || test_label_path == NULL){
        fprintf(stderr, "Creating data path failed.\n");
    }

    strcpy(train_path, data_dir);
    strcpy(test_path, data_dir);
    strcpy(train_label_path, data_dir);
    strcpy(test_label_path, data_dir);
    strcat(train_path, train_folder_name);
    strcat(test_path, test_folder_name);
    strcat(train_label_path, train_folder_name);
    strcat(test_label_path, test_folder_name);
    strcat(train_label_path, label_name);
    strcat(test_label_path, label_name);

    int width = model->config.image_width;
    int height = model->config.image_height;
    int channels = model->config.channels;
    BGR** allPixels_train = NULL;
    BGR** allPixels_test = NULL;

    // Reading the train data
    if(ReadAllBMPsInDirectory(train_path, &allPixels_train, model->nImages, width, height)==0){
        printf("successfully read all the BMP files - train data.\n");
        if(LabelReader(train_label_path, model->nImages, &(model->labels_train))==0){
            printf("successfully read all the labels - train data.\n");
        }
    }

    // Reading the test data
    if(ReadAllBMPsInDirectory(test_path, &allPixels_test, model->nImages_test, width, height)==0){
        printf("successfully read all the BMP files - test data.\n");
        if(LabelReader(test_label_path, model->nImages_test, &(model->labels_test))==0){
            printf("successfully read all the labels - test data.\n");
        }
    }

    // Shuffle train (image/label) pairs
    ShuffleData(allPixels_train, model->labels_train, model->nImages, 42);

    // Linearize train and test dataset
    ConvertTo1DFloatArray(allPixels_train, model->nImages, width, height, channels, &(model->data_train));
    ConvertTo1DFloatArray(allPixels_test, model->nImages_test, width, height, channels, &(model->data_test));

    // Deallocate temporary data defined within the function.
    for(int i=0; i<model->nImages; ++i){
        free(allPixels_train[i]);
    }
    for(int i=0; i<model->nImages_test; ++i){
        free(allPixels_test[i]);
    }
    free(allPixels_train);
    free(allPixels_test);
    free(train_path);
    free(test_path);
    free(train_label_path);
    free(test_label_path);
}

void GetBatch(ViTModel* model, float* batch_data, int* batch_labels){
    int start_b_idx = model->curr_batch_idx;
    int end_b_idx = start_b_idx + model->batch_size > model->nImages ? model->nImages : start_b_idx + model->batch_size;

    int channels = model->config.channels;
    int height = model->config.image_height;
    int width = model->config.image_width;

    for(int b=start_b_idx; b<end_b_idx; ++b){
        int data_idx = b*channels*height*width;
        int batch_idx = (b-start_b_idx)*channels*height*width;
        memcpy(&batch_data[batch_idx], &(model->data_train[data_idx]), channels * width * height * sizeof(float));
        batch_labels[b-start_b_idx] = model->labels_train[b];
    }
}

void ViT_from_YAML(ViTModel* model, const char* yaml_path){

    FILE* file = fopen(yaml_path, "r");
    if(!file){
        perror("Unable to open YAML file");
        return;
    }

    char line[256];
    while(fgets(line, sizeof(line), file)){
        char* trimmed_line = trim_whitespace(line);
        if(strncmp(trimmed_line, "image_width:", 12)==0){
            model->config.image_width = parse_int_value(trimmed_line);
        } else if (strncmp(trimmed_line, "image_height:", 13) == 0) {
            model->config.image_height = parse_int_value(trimmed_line);
        } else if (strncmp(trimmed_line, "channels:", 9) == 0) {
            model->config.channels = parse_int_value(trimmed_line);
        } else if (strncmp(trimmed_line, "patch_size:", 11) == 0) {
            model->config.patch_size = parse_int_value(trimmed_line);
        } else if (strncmp(trimmed_line, "hidden_size:", 12) == 0) {
            model->config.hidden_size = parse_int_value(trimmed_line);
        } else if (strncmp(trimmed_line, "num_attention_heads:", 20) == 0) {
            model->config.num_attention_heads = parse_int_value(trimmed_line);
        } else if (strncmp(trimmed_line, "num_layers:", 11) == 0) {
            model->config.num_layers = parse_int_value(trimmed_line);
        } else if (strncmp(trimmed_line, "num_classes:", 12) == 0) {
            model->config.num_classes = parse_int_value(trimmed_line);
        } else if (strncmp(trimmed_line, "batch_size:", 11) == 0) {
            model->batch_size = parse_int_value(trimmed_line);
        } 
    }
}