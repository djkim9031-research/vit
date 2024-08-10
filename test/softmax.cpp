#include <gtest/gtest.h>
#include "softmax.h"

// softmax forward call.
TEST(SoftmaxTest, forward_call){
    // (2, 1, 10)
    float logits[20] = {0.01, -0.1, 0.2, 0.3, -0.4, 0.31, 0.12, -0.15, 0.9, 0.2,
                        -0.8, -0.4, 0.12, 0.76, 0.14, -0.27, 0.3, 0.1, 0.5, 0.2};
    
    float probs[20] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // probs_truth is pytorch tensor output.
    /* Python code

    torch.set_printoptions(precision=6)
    logits = [0.01, -0.1, 0.2, 0.3, -0.4, 0.31, 0.12, -0.15, 0.9, 0.2,
              -0.8, -0.4, 0.12, 0.76, 0.14, -0.27, 0.3, 0.1, 0.5, 0.2]
    logits = torch.tensor(logits, dtype=torch.float32).view(2, 1, 10)
    logits = logits.requires_grad_()

    probs = nn.functional.softmax(logits, dim=-1)

    print(probs)
    */

    float probs_truth[20] = {0.082865, 0.074233, 0.100204, 0.110743, 0.054993, 0.111856, 0.092500, 0.070613, 0.201787, 0.100204,
                             0.038655, 0.057666, 0.096995, 0.183950, 0.098955, 0.065671, 0.116125, 0.095075, 0.141835, 0.105074};
    float tolerance = 5e-6;
    softmax_forward(logits, probs, 2, 10);
    for(int i=0; i<20; ++i){
        EXPECT_NEAR(probs[i], probs_truth[i], tolerance);
    }
}

// softmax-crossentropy forward call.
TEST(SoftmaxCrossentropyTest, forward_call){
    // (2, 1, 10)
    float logits[20] = {0.01, -0.1, 0.2, 0.3, -0.4, 0.31, 0.12, -0.15, 0.9, 0.2,
                        -0.8, -0.4, 0.12, 0.76, 0.14, -0.27, 0.3, 0.1, 0.5, 0.2};
    
    float probs[20] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // (2, 1, 1)
    int targets[2] = {8, 8};
    float losses[2] = {0.0, 0.0};

    // losses_truth is pytorch tensor output.
    /* Python code

    import torch
    from torch import nn
    import torch.nn.functional as F

    torch.set_printoptions(precision=6)

    # Define logits and targets
    logits = [0.01, -0.1, 0.2, 0.3, -0.4, 0.31, 0.12, -0.15, 0.9, 0.2,
            -0.8, -0.4, 0.12, 0.76, 0.14, -0.27, 0.3, 0.1, 0.5, 0.2]
    logits = torch.tensor(logits, dtype=torch.float32).view(2, 10)
    logits = logits.requires_grad_()

    targets = [8, 8]
    targets = torch.tensor(targets, dtype=torch.long)

    # Custom Cross-Entropy loss function
    def custom_cross_entropy_loss(logits, targets):
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs)

        target_log_probs = -log_probs[range(logits.size(0)), targets]

        return target_log_probs

    loss = custom_cross_entropy_loss(logits, targets)

    print("loss: ", loss)
    */

    float losses_truth[2] = {1.600543, 1.953092};
    float tolerance = 5e-6;

    softmax_forward(logits, probs, 2, 10);
    crossentropy_forward(probs, targets, losses, 2, 10);
    for(int i=0; i<2; ++i){
        EXPECT_NEAR(losses[i], losses_truth[i], tolerance);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}