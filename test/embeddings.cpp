#include <gtest/gtest.h>
#include "embeddings.h"
#include "conv2d.h"

// Matmul forward call test function.
TEST(EmbeddingsTest, forward_call) {
    // Creating patch embedding
    // 108 = (B=1, C=3, H=6, W=6)
    float conv_x[108] = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                         2.0, 3.0, 4.0, 5.0, 1.0, 2.2, 3.0, 4.0, 5.0,
                         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0,
                         2.0, 3.0, 4.0, 5.0, 1.0, 2.9, 3.0, 4.0, 5.0,
                         1.0, 1.0, 2.0, 3.7, 2.5, 2.5, 3.0, 4.0, 5.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.1, 4.0, 5.0,
                         1.0, 1.0, 1.2, 1.0, 1.0, 2.0, 3.1, 4.0, 5.0,
                         2.0, 3.0, 4.0, 5.0, 1.0, 2.5, 3.0, 4.0, 5.0,
                         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0,
                         2.0, 3.0, 4.0, 5.9, 1.0, 2.0, 3.0, 4.0, 5.0,
                         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.4, 4.0, 5.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    // 12 = (out channel = 1, in channel = 3, kernel H = 2, kernel W = 2)
    float kernel[12] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                        0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
    // 1 = (out channel = 1)
    float bias[1] = {0.65};
    // 9 = (B=1, out channel = 1, out H = 3, out W = 3)
    float conv_y[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // class token
    // 1 = (1, 1, 1)
    float cls_tok[1] = {0.3};

    // position embedding
    // 10 = (1, 9+1, 1)
    float pos_embd[10] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    float z[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // z_truth is pytorch tensor output.
    /* Python code

    torch.set_printoptions(precision=5)
    x = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
         2.0, 3.0, 4.0, 5.0, 1.0, 2.2, 3.0, 4.0, 5.0,
         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0,
         2.0, 3.0, 4.0, 5.0, 1.0, 2.9, 3.0, 4.0, 5.0,
         1.0, 1.0, 2.0, 3.7, 2.5, 2.5, 3.0, 4.0, 5.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.1, 4.0, 5.0,
         1.0, 1.0, 1.2, 1.0, 1.0, 2.0, 3.1, 4.0, 5.0,
         2.0, 3.0, 4.0, 5.0, 1.0, 2.5, 3.0, 4.0, 5.0,
         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0,
         2.0, 3.0, 4.0, 5.9, 1.0, 2.0, 3.0, 4.0, 5.0,
         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.4, 4.0, 5.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    x = torch.tensor(x, dtype=torch.float32).view(1, 3, 6, 6)
    x = x.requires_grad_()

    kernel = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
              0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    kernel = torch.tensor(kernel, requires_grad=True)
    kernel = nn.Parameter(kernel.view(1, 3, 2, 2))
    bias = nn.Parameter(torch.tensor([0.65], requires_grad=True))
    conv = nn.Conv2d(3, 1, (2,2), stride=2)
    conv.weight = kernel
    conv.bias = bias
    y = conv(x)
    y = y.flatten(2).transpose(1,2)

    cls_token = nn.Parameter(torch.tensor([0.3], requires_grad=True).view(1, 1, 1))
    pos_embd = nn.Parameter(torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], requires_grad=True).view(1,10,1))

    z = torch.cat((cls_token, y), dim=1)
    z = z + pos_embd
    print(z)
    */
    float z_truth[10] = {0.30000, 6.15000, 7.39000, 6.75000, 5.23000, 6.65000, 9.25000, 8.45000, 8.73000, 10.15000};
    float tolerance = 5e-6;

    conv2d_forward(conv_x, kernel, bias, conv_y, 1, 3, 6, 6, 1, 2, 2, 2);
    embeddings_forward(conv_y, cls_tok, pos_embd, z, 1, 9, 1, 3, 3);
    for(int i=0; i<10; ++i){
        EXPECT_NEAR(z[i], z_truth[i], tolerance);
    }
}


// Matmul backward call test function.
TEST(EmbeddingsTest, backward_call) {
    // Creating patch embedding
    // 108 = (B=1, C=3, H=6, W=6)
    float conv_x[108] = {1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                         2.0, 3.0, 4.0, 5.0, 1.0, 2.2, 3.0, 4.0, 5.0,
                         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0,
                         2.0, 3.0, 4.0, 5.0, 1.0, 2.9, 3.0, 4.0, 5.0,
                         1.0, 1.0, 2.0, 3.7, 2.5, 2.5, 3.0, 4.0, 5.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.1, 4.0, 5.0,
                         1.0, 1.0, 1.2, 1.0, 1.0, 2.0, 3.1, 4.0, 5.0,
                         2.0, 3.0, 4.0, 5.0, 1.0, 2.5, 3.0, 4.0, 5.0,
                         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0,
                         2.0, 3.0, 4.0, 5.9, 1.0, 2.0, 3.0, 4.0, 5.0,
                         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.4, 4.0, 5.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    // 12 = (out channel = 1, in channel = 3, kernel H = 2, kernel W = 2)
    float kernel[12] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                        0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
    // 1 = (out channel = 1)
    float bias[1] = {0.65};
    // 9 = (B=1, out channel = 1, out H = 3, out W = 3)
    float conv_y[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // class token
    // 1 = (1, 1, 1)
    float cls_tok[1] = {0.3};

    // position embedding
    // 10 = (1, 9+1, 1)
    float pos_embd[10] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    float z[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // derivatives to calculate
    float dz[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    float dconv_y[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float dcls_tok[1] = {0.0};
    float dpos_embd[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    float dconv_x[108] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float dkernel[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float dbias[1] = {0.0};

    // truth values are pytorch tensor.grad outputs.
    /* Python code

    torch.set_printoptions(precision=5)
    x = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0,
         2.0, 3.0, 4.0, 5.0, 1.0, 2.2, 3.0, 4.0, 5.0,
         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0,
         2.0, 3.0, 4.0, 5.0, 1.0, 2.9, 3.0, 4.0, 5.0,
         1.0, 1.0, 2.0, 3.7, 2.5, 2.5, 3.0, 4.0, 5.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.1, 4.0, 5.0,
         1.0, 1.0, 1.2, 1.0, 1.0, 2.0, 3.1, 4.0, 5.0,
         2.0, 3.0, 4.0, 5.0, 1.0, 2.5, 3.0, 4.0, 5.0,
         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.0, 4.0, 5.0,
         2.0, 3.0, 4.0, 5.9, 1.0, 2.0, 3.0, 4.0, 5.0,
         1.0, 1.0, 2.0, 3.0, 2.5, 2.5, 3.4, 4.0, 5.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    x = torch.tensor(x, dtype=torch.float32).view(1, 3, 6, 6)
    x = x.requires_grad_()

    kernel = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
              0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    kernel = torch.tensor(kernel, requires_grad=True)
    kernel = nn.Parameter(kernel.view(1, 3, 2, 2))
    bias = nn.Parameter(torch.tensor([0.65], requires_grad=True))
    conv = nn.Conv2d(3, 1, (2,2), stride=2)
    conv.weight = kernel
    conv.bias = bias
    y_conv = conv(x)
    y_conv.retain_grad()
    y = y_conv.flatten(2).transpose(1,2)

    cls_token = nn.Parameter(torch.tensor([0.3], requires_grad=True).view(1, 1, 1))
    pos_embd = nn.Parameter(torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], requires_grad=True).view(1,10,1))

    z = torch.cat((cls_token, y), dim=1)
    z = z + pos_embd

    z.sum().backward()
    print(pos_embd.grad)
    print(cls_token.grad)
    print(y_conv.grad)
    print(kernel.grad)
    print(bias.grad)
    print(x.grad)
    */
    float dpos_embd_truth[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    float dcls_tok_truth[1] = {1.0};
    float dconv_y_truth[9] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    float dkernel_truth[12] = {25.20000, 23.00000, 28.40000, 25.50000, 23.60000, 26.30000, 
                               23.70000, 19.00000, 26.80000, 21.50000, 23.50000, 25.50000};
    float dbias_truth[1] = {9.0};
    float dconv_x_truth[108] = {0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000,
                                0.20000, 0.20000, 0.20000, 0.20000, 0.20000, 0.20000};
    float tolerance = 5e-6;

    conv2d_forward(conv_x, kernel, bias, conv_y, 1, 3, 6, 6, 1, 2, 2, 2);
    embeddings_forward(conv_y, cls_tok, pos_embd, z, 1, 9, 1, 3, 3);
    embeddings_backward(conv_y, cls_tok, pos_embd, dconv_y, dcls_tok, dpos_embd, dz, 1, 9, 1, 3, 3);
    conv2d_backward(conv_x, kernel, dconv_x, dkernel, dbias, dconv_y, 1, 3, 6, 6, 1, 2, 2, 2);
    for(int i=0;i<10;++i){
        EXPECT_NEAR(dpos_embd[i], dpos_embd_truth[i], tolerance);
    }
    EXPECT_NEAR(dcls_tok[0], dcls_tok_truth[0], tolerance);
    for(int i=0; i<9;++i){
        EXPECT_NEAR(dconv_y[i], dconv_y_truth[i], tolerance);
    }
    for(int i=0; i<12;++i){
        EXPECT_NEAR(dkernel[i], dkernel_truth[i], tolerance);
    }
    EXPECT_NEAR(dbias[0], dbias_truth[0], tolerance);
    for(int i=0; i<108;++i){
        EXPECT_NEAR(dconv_x[i], dconv_x_truth[i], tolerance);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}