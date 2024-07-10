#include <gtest/gtest.h>
#include "conv2d.h"
#include "residual.h"
#include "activations.h"

// Conv2d connection, forward call.
TEST(Conv2dTest, forward_call) {
    // 27 = (B=1, C=3, H=3, W=3)
    float conv_x[27] = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                        2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                        1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0};
    // 12 = (out channel = 1, in channel = 3, kernel H = 2, kernel W = 2)
    float kernel[12] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                        0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
    // 1 = (out channel = 1)
    float bias[1] = {0.65};
    // 4 = (B=1, out channel = 1, out H = 2, out W = 2)
    float conv_y[4] = {0.0, 0.0, 0.0, 0.0};
    
    // conv_y_truth is pytorch tensor output.
    /* Python code

    torch.set_printoptions(precision=5)
    x = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
         2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0]
    x = torch.tensor(x, dtype=torch.float32).view(1, 3, 3, 3)
    x = x.requires_grad_()

    kernel = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
              0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    kernel = torch.tensor(kernel, requires_grad=True)
    kernel = nn.Parameter(kernel.view(1, 3, 2, 2))
    bias = nn.Parameter(torch.tensor([0.65], requires_grad=True))
    conv = nn.Conv2d(3, 1, (2,2))

    conv.weight = kernel
    conv.bias = bias

    y = conv(x)
    print(y)
    */
    float conv_y_truth[4] = {5.15000, 5.25000, 7.55000, 8.25000};
    float tolerance = 5e-6;

    conv2d_forward(conv_x, kernel, bias, conv_y, 1, 3, 3, 3, 1, 2, 2);
    for(int i=0; i<4; ++i){
        EXPECT_NEAR(conv_y[i], conv_y_truth[i], tolerance);
    }
}

// Conv2d connection, backward call.
TEST(Conv2dTest, backward_call) {
    // 27 = (B=1, C=3, H=3, W=3)
    float conv_x[27] = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                        2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                        1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0};
    // 12 = (out channel = 1, in channel = 3, kernel H = 2, kernel W = 2)
    float kernel[12] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                        0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
    // 1 = (out channel = 1)
    float bias[1] = {0.65};
    // 4 = (B=1, out channel = 1, out H = 2, out W = 2)
    float conv_y[4] = {0.0, 0.0, 0.0, 0.0};

    // derivates to populate
    float conv_dx[27] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float dkernel[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float dbias[1] = {0.0};
    float conv_dy[4] = {1.0, 1.0, 1.0, 1.0};

    // truth values are pytorch tensor.grad outputs.
    /* Python code

    torch.set_printoptions(precision=5)
    x = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
         2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0]
    x = torch.tensor(x, dtype=torch.float32).view(1, 3, 3, 3)
    x = x.requires_grad_()

    kernel = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
              0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    kernel = torch.tensor(kernel, requires_grad=True)
    kernel = nn.Parameter(kernel.view(1, 3, 2, 2))
    bias = nn.Parameter(torch.tensor([0.65], requires_grad=True))
    conv = nn.Conv2d(3, 1, (2,2))

    conv.weight = kernel
    conv.bias = bias

    y = conv(x)
    y.sum().backward()
    print(x.grad)
    print(kernel.grad)
    print(bias.grad)
    */

    float conv_dx_truth[27] = {0.20000, 0.40000, 0.20000, 0.40000, 0.80000, 0.40000, 0.20000, 0.40000, 0.20000,
                               0.20000, 0.40000, 0.20000, 0.40000, 0.80000, 0.40000, 0.20000, 0.40000, 0.20000,
                               0.20000, 0.40000, 0.20000, 0.40000, 0.80000, 0.40000, 0.20000, 0.40000, 0.20000};
    float dkernel_truth[12] = {4.00000,  5.00000, 9.00000, 12.00000,
                               11.00000, 10.00000, 13.00000, 12.00000,
                               7.50000,  8.00000, 12.50000, 14.00000};
    float dbias_truth[1] = {4.00000};
    float tolerance = 5e-6;

    conv2d_forward(conv_x, kernel, bias, conv_y, 1, 3, 3, 3, 1, 2, 2);
    conv2d_backward(conv_x, kernel, conv_dx, dkernel, dbias, conv_dy, 1, 3, 3, 3, 1, 2, 2);
    for(int i=0; i<27; ++i){
        EXPECT_NEAR(conv_dx[i], conv_dx_truth[i], tolerance);
    }
    for(int i=0; i<12; ++i){
        EXPECT_NEAR(dkernel[i], dkernel_truth[i], tolerance);
    }
    EXPECT_NEAR(dbias[0], dbias_truth[0], tolerance);

}

// Conv2d connection, residual, then gelu test.
TEST(nn_with_conv2d_resi_gelu, forward_backward_call) {

    // Conv2d tensors and derivate tensors.
    // 27 = (B=1, C=3, H=3, W=3)
    float conv_x[27] = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                        2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                        1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0};
    // 12 = (out channel = 1, in channel = 3, kernel H = 2, kernel W = 2)
    float kernel[12] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                        0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
    // 1 = (out channel = 1)
    float bias[1] = {0.65};
    // 4 = (B=1, out channel = 1, out H = 2, out W = 2)
    float conv_y[4] = {0.0, 0.0, 0.0, 0.0};

    // derivates to populate
    float conv_dx[27] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float dkernel[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float dbias[1] = {0.0};
    float conv_dy[4] = {0.0, 0.0, 0.0, 0.0};

    // Residual tensors and derivative tensors.
    float skip_input[4] = {3.4, 2.4, 1.2, 0.5};
    float dskip[4] = {0.0, 0.0, 0.0, 0.0};
    float residual_output[4] = {0.0, 0.0, 0.0, 0.0};
    float dresidual[4] = {0.0, 0.0, 0.0, 0.0};

    // gelu tensors and derivative tensors.
    float gelu_output[4] = {0.0, 0.0, 0.0, 0.0};
    float dgelu[4] = {1.0, 1.0, 1.0, 1.0};

    // truth values are pytorch tensor.grad outputs.
    /* Python code

    torch.set_printoptions(precision=5)
    x = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
         2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0,
         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0]
    x = torch.tensor(x, dtype=torch.float32).view(1, 3, 3, 3)
    x = x.requires_grad_()

    kernel = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
              0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    kernel = torch.tensor(kernel, requires_grad=True)
    kernel = nn.Parameter(kernel.view(1, 3, 2, 2))
    bias = nn.Parameter(torch.tensor([0.65], requires_grad=True))
    conv = nn.Conv2d(3, 1, (2,2))

    conv.weight = kernel
    conv.bias = bias

    y = conv(x).view(-1)
    y = y.requires_grad_()

    skip_input = [3.4, 2.4, 1.2, 0.5]
    skip_input = torch.tensor(skip_input).requires_grad_()

    residual_output = (skip_input + y)/10
    gelu = GELUActivation()
    gelu_output = gelu(residual_output)

    print(gelu_output)
    gelu_output.sum().backward()
    print(x.grad)
    print(kernel.grad)
    print(bias.grad)
    */

    float gelu_output_truth[4] = {0.68708, 0.59499, 0.70795, 0.70795};
    float conv_dx_truth[27] = {0.02080, 0.04091, 0.02011, 0.04174, 0.08278, 0.04105, 0.02094, 0.04188, 0.02094,
                               0.02080, 0.04091, 0.02011, 0.04174, 0.08278, 0.04105, 0.02094, 0.04188, 0.02094,
                               0.02080, 0.04091, 0.02011, 0.04174, 0.08278, 0.04105, 0.02094, 0.04188, 0.02094};
    float dkernel_truth[12] = {0.41392, 0.51861, 0.93738, 1.24730,
                               1.13776, 1.02823, 1.35340, 1.24730,
                               0.78034, 0.82853, 1.29619, 1.45357};
    float dbias_truth[1] = {0.41392};


    float tolerance = 5e-6;

    conv2d_forward(conv_x, kernel, bias, conv_y, 1, 3, 3, 3, 1, 2, 2);
    residual_forward(skip_input, conv_y, residual_output, 4);
    for(int i=0; i<4; ++i){
        residual_output[i]/=10.f; // Since we added large skip inputs, normalize it by 10 will prevent gelu saturation. (grad = 1)
    }
    gelu_forward(residual_output, gelu_output, 4);
    gelu_backward(residual_output, dresidual, dgelu, 4);
    for(int i=0; i<4; ++i){
        dresidual[i]/=10.f; // From chain rule, since we normalized residual_output by 10, its derivative of 1/10 is applied in the chain.
    }
    residual_backward(dskip, conv_dy, dresidual, 4);
    conv2d_backward(conv_x, kernel, conv_dx, dkernel, dbias, conv_dy, 1, 3, 3, 3, 1, 2, 2);

    for(int i=0; i<4; ++i){
        EXPECT_NEAR(gelu_output[i], gelu_output_truth[i], tolerance);
    }
    for(int i=0; i<27; ++i){
        EXPECT_NEAR(conv_dx[i], conv_dx_truth[i], tolerance);
    }
    for(int i=0; i<12; ++i){
        EXPECT_NEAR(dkernel[i], dkernel_truth[i], tolerance);
    }
    EXPECT_NEAR(dbias[0], dbias_truth[0], tolerance);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}