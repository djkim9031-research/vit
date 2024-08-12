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