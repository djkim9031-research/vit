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