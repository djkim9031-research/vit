#include <gtest/gtest.h>
#include "activations.h"

// GeLU forward call test function.
TEST(GeLUTest, forward_call) {
    float x[5] = {0.0, 0.1, 0.2, 0.3, 0.4};
    float y[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    // y_truth is pytorch tensor output.
    /* Python code

    torch.set_printoptions(precision=7)
    n = GELUActivation()
    x = [0.0, 0.1, 0.2, 0.3, 0.4]
    x = torch.tensor(x, requires_grad=True)
    y = n(x)
    print(y)
    */
    float y_truth[5] = {0.0000000, 0.0539828, 0.1158514, 0.1853709, 0.2621612};
    float tolerance = 5e-7;

    gelu_forward(x, y, 5);
    for(int i=0; i<5; ++i){
        EXPECT_NEAR(y[i], y_truth[i], tolerance);
    }
}

// GeLU backward call test function.
TEST(GeLUTest, backward_call){
    float x[5] = {0.0, 0.1, 0.2, 0.3, 0.4};
    float y[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    float dx[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    float dy[5] = {1.0, 1.0, 1.0, 1.0, 1.0};

    // dx_truth is pytorch tensor.grad output.
    /* Python code

    torch.set_printoptions(precision=7)
    n = GELUActivation()
    x = [0.0, 0.1, 0.2, 0.3, 0.4]
    x = torch.tensor(x, requires_grad=True)
    y = n(x)
    y.sum().backward()
    print(x.grad)
    */
   float dx_truth[5] = {0.5000000, 0.5795218, 0.6574581, 0.7322955, 0.8026589};
   float tolerance = 5e-7;

   gelu_forward(x, y, 5);
   gelu_backward(x, dx, dy, 5);
   for(int i=0; i<5; ++i){
        EXPECT_NEAR(dx[i], dx_truth[i], tolerance);
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}