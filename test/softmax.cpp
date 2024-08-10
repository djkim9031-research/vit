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

// softmax-crossentropy forward call with the custom python crossentropy function 
TEST(SoftmaxCrossEntropyTest_customCEfunction, forward_call){
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

// softmax-crossentropy forward call with the torch crossentropy function 
TEST(SoftmaxCrossEntropyTest_torchCEfunction, forward_call){
    // (2, 1, 10)
    float logits[20] = {0.01, -0.1, 0.2, 0.3, -0.4, 0.31, 0.12, -0.15, 0.9, 0.2,
                        -0.8, -0.4, 0.12, 0.76, 0.14, -0.27, 0.3, 0.1, 0.5, 0.2};
    
    float probs[20] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // (2, 1, 1)
    int targets[2] = {8, 8};
    float losses[2] = {0.0, 0.0};

    // loss_truth is pytorch tensor output.
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

    loss = torch.nn.functional.cross_entropy(logits, targets)
    print(loss)
    */

    softmax_forward(logits, probs, 2, 10);
    crossentropy_forward(probs, targets, losses, 2, 10);
    float mean_loss = (losses[0] + losses[1])/2.f;
    float loss_truth = 1.776817;
    float tolerance = 5e-6;

    EXPECT_NEAR(mean_loss, loss_truth, tolerance);   
}

// softmax-crossentropy backward call
TEST(SoftmaxCrossEntropyTest, backward_call){
    // (2, 1, 10)
    float logits[20] = {0.01, -0.1, 0.2, 0.3, -0.4, 0.31, 0.12, -0.15, 0.9, 0.2,
                        -0.8, -0.4, 0.12, 0.76, 0.14, -0.27, 0.3, 0.1, 0.5, 0.2};
    
    float probs[20] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // (2, 1, 1)
    int targets[2] = {8, 8};
    float losses[2] = {0.0, 0.0};

    // derivatives
    float dlogits[20] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float dlosses[2] = {0.5, 0.5}; // (1.0 / batch_size)

    // dlogits_truth is pytorch tensor output
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

    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    print(logits.grad)
    
    */

    float dlogits_truth[20] = {0.041433, 0.037117, 0.050102, 0.055372, 0.027497, 0.055928, 0.046250, 0.035306, -0.399107, 0.050102,
                               0.019327, 0.028833, 0.048498, 0.091975, 0.049477, 0.032836, 0.058062, 0.047537, -0.429083, 0.052537};
    float tolerance = 5e-6;

    softmax_forward(logits, probs, 2, 10);
    crossentropy_forward(probs, targets, losses, 2, 10);
    crossentropy_softmax_backward(probs, targets, dlogits, dlosses, 2, 10);

    for(int i=0; i<20; ++i){
        EXPECT_NEAR(dlogits[i], dlogits_truth[i], tolerance);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}