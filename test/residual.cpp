#include <gtest/gtest.h>
#include "residual.h"
#include "activations.h"

// Residual connection, then GeLU nonlinearity forward call.
TEST(ResidualTest, forward_call) {
    float x[5] = {0.0, 0.1, 0.2, 0.3, 0.4};
    float y[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    float z[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    // z_truth is pytorch tensor output.
    /* Python code

    torch.set_printoptions(precision=7)
    n = GELUActivation()
    x = [0.0, 0.1, 0.2, 0.3, 0.4]
    x = torch.tensor(x, requires_grad=True)
    y = n(x)
    z = y + x
    print(z)
    */
    float z_truth[5] = {0.0000000, 0.1539828, 0.3158514, 0.4853709, 0.6621612};
    float tolerance = 5e-7;

    gelu_forward(x, y, 5);
    residual_forward(y, x, z, 5);
    for(int i=0; i<5; ++i){
        EXPECT_NEAR(z[i], z_truth[i], tolerance);
    }
}

// Residual connection, then GeLU nonlinearity backward call.
TEST(ResidualTest, backward_call) {
    float x[5] = {0.0, 0.1, 0.2, 0.3, 0.4};
    float y[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    float z[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    float dx[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    float dy[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    float dz[5] = {1.0, 1.0, 1.0, 1.0, 1.0};

    // dx_truth is pytorch tensor.grad output.
    /* Python code

    torch.set_printoptions(precision=7)
    n = GELUActivation()
    x = [0.0, 0.1, 0.2, 0.3, 0.4]
    x = torch.tensor(x, requires_grad=True)
    y = n(x)
    z = y + x
    z.sum().backward()
    print(x.grad)
    */

   float dx_truth[5] = {1.5000000, 1.5795219, 1.6574581, 1.7322955, 1.8026588};
   float tolerance = 5e-7;
   gelu_forward(x, y, 5);
   residual_forward(y, x, z, 5);
   residual_backward(dy, dx, dz, 5);
   gelu_backward(x, dx, dy, 5);
   for(int i=0; i<5; ++i){
        EXPECT_NEAR(dx[i], dx_truth[i], tolerance);
    }
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}