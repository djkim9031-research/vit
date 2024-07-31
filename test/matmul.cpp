#include <gtest/gtest.h>
#include "matmul.h"

// Matmul forward call test function.
TEST(MatmulTest, forward_call) {
    // (1, 3, 3)
    float x[9] = {0.0, 1.1, 0.2, 3.3, 2.4, 1.5, 0.6, 2.7, 0.8};
    // (3, 2)
    float weight[6] = {0.31, 0.2, 0.154, 0.731, 0.12, 0.9};
    // (2)
    float bias[2] = {0.293, 0.749};
    // (1, 3, 2)
    float y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // y_truth is pytorch tensor output.
    /* Python code

    torch.set_printoptions(precision=5)
    x = [0.0, 1.1, 0.2, 3.3, 2.4, 1.5, 0.6, 2.7, 0.8]
    x = torch.tensor(x, requires_grad=True)
    x = torch.tensor(x, dtype=torch.float32).view(1, 3, 3)
    x = x.requires_grad_()
    weight = torch.tensor([0.31, 0.2, 0.154, 0.731, 0.12, 0.9], requires_grad=True)
    weight = nn.Parameter(weight.view(3, 2))
    bias = torch.tensor([0.293, 0.749], requires_grad=True)
    bias = nn.Parameter(bias.view(2))

    y = x@weight + bias
    print(y)
    */
    float y_truth[6] = {0.48640, 1.73310, 1.86560, 4.51340, 0.99080, 3.56270};
    float tolerance = 5e-6;

    matmul_forward(x, y, weight, bias, 1, 3, 3, 2);
    for(int i=0; i<6; ++i){
        EXPECT_NEAR(y[i], y_truth[i], tolerance);
    }
}

// Matmul backward call test function.
TEST(MatmulTest, backward_call){
    // (1, 3, 3)
    float x[9] = {0.0, 1.1, 0.2, 3.3, 2.4, 1.5, 0.6, 2.7, 0.8};
    // (3, 2)
    float weight[6] = {0.31, 0.2, 0.154, 0.731, 0.12, 0.9};
    // (2)
    float bias[2] = {0.293, 0.749};
    // (1, 3, 2)
    float y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    float dx[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float dweight[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float dbias[2] = {0.0, 0.0};
    float dy[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    // dx_truth is pytorch tensor.grad output.
    /* Python code

    torch.set_printoptions(precision=5)
    x = [0.0, 1.1, 0.2, 3.3, 2.4, 1.5, 0.6, 2.7, 0.8]
    x = torch.tensor(x, requires_grad=True)
    x = torch.tensor(x, dtype=torch.float32).view(1, 3, 3)
    x = x.requires_grad_()
    weight = torch.tensor([0.31, 0.2, 0.154, 0.731, 0.12, 0.9], requires_grad=True)
    weight = nn.Parameter(weight.view(3, 2))
    bias = torch.tensor([0.293, 0.749], requires_grad=True)
    bias = nn.Parameter(bias.view(2))

    y = x@weight + bias
    y.sum().backward()
    print(weight.grad)
    print(bias.grad)
    print(x.grad)
    */
    float dx_truth[9] = {0.51000, 0.88500, 1.02000, 0.51000, 0.88500, 1.02000, 0.51000, 0.88500, 1.02000};
    float dweight_truth[6] = {3.90000, 3.90000, 6.20000, 6.20000, 2.50000, 2.50000};
    float dbias_truth[2] = {3.0, 3.0};
    float tolerance = 5e-6;

    matmul_forward(x, y, weight, bias, 1, 3, 3, 2);
    matmul_backward(x, weight, dx, dweight, dbias, dy, 1, 3, 3, 2);
    for(int i=0; i<6; ++i){
        EXPECT_NEAR(dweight[i], dweight_truth[i], tolerance);
    }
    for(int i=0; i<2; ++i){
        EXPECT_NEAR(dbias[i], dbias_truth[i], tolerance);
    }
    for(int i=0; i<9; ++i){
        EXPECT_NEAR(dx[i], dx_truth[i], tolerance);
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}