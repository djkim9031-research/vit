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
    printf("Allocating %d MiB for parameter gradients.\n", (int)round(model->num_parameters * sizeof(floatX)/(1024*1024)));
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
    printf("Allocating %zd MiB for activation/gradients tensors.\n", num_act_bytes/(1024*1024));

    // Create memory for cahcing inputs and targets
    cudaCheck(cudaMalloc((void**)&model->inputs, B*im_C*im_H*im_W*sizeof(float)));
    cudaCheck(cudaMalloc((void**)&model->targets, B*sizeof(float)));
    cudaCheck(cudaMalloc((void**)&model->accumulated_mean_loss, sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&model->cpu_loss, sizeof(float)));

    // AdamW optimizer parameters.
    printf("Allocating %zu MiB for AdamW optimizer state m.\n", sizeof(float)>>20);
    printf("Allocating %zu MiB for AdamW optimizer state v.\n", sizeof(float)>>20);
    assert(model->m_memory == nullptr);
    assert(model->v_memory == nullptr);
    cudaCheck(cudaMalloc((void**)&model->m_memory, sizeof(float)));
    cudaCheck(cudaMalloc((void**)&model->v_memory, sizeof(float)));

    // Memory usage info
    size_t free, total;
    cudaCheck(cudaMemGetInfo(&free, &total));
    printf("[INFO] device memory usage %zd MiB / %zd MiB.\n", (total-free)/(1024*1024), total/(1024*1024));
}