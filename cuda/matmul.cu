#include "matmul.cuh"


// -----------------------------------------------------------------------------------------
// GPU kernels

__global__ void matmul_forward_kernel1(floatX* x, floatX* y, floatX* weight, floatX* bias,
                                       int B_in_r, int in_c, int ou_c){
    
    int B_in_r_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int ou_c_dim = blockIdx.y * blockDim.y + threadIdx.y;
    if(B_in_r_dim < B_in_r && ou_c_dim < ou_c){
        floatX val = (bias != NULL) ? bias[ou_c_dim] : 0.f;
        for(int ic = 0; ic<in_c; ++ic){
            val += x[B_in_r_dim*in_c + ic] * weight[ic*ou_c + ou_c_dim];
        }

        y[B_in_r_dim*ou_c + ou_c_dim] = val;
    }
}

__global__ void matmul_backward_kernel1(floatX* x, floatX* weight, floatX* dx, floatX* dweight, floatX* dbias,
                                        floatX* dy, int B_in_r, int in_c, int ou_c){
    
    int B_in_r_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int ou_c_dim = blockIdx.y * blockDim.y + threadIdx.y;

    if(B_in_r_dim < B_in_r && ou_c_dim < ou_c){
        floatX grad = dy[B_in_r_dim*ou_c + ou_c_dim];

        if(dbias != NULL){
            atomicAdd(&dbias[ou_c_dim], grad);
        }

        for(int ic = 0; ic < in_c; ++ic){
            atomicAdd(&dx[B_in_r_dim*in_c + ic], grad * weight[ic*ou_c + ou_c_dim]);
            atomicAdd(&dweight[ic*ou_c + ou_c_dim], grad * x[B_in_r_dim*in_c + ic]);
        }
    }
}

// Helper function to extract a tensor at index t in sequence T of the original tensor.
// This is used for modified matmul function, which will be called before the classfication stage
// of ViT.
// 
// @param orig          linearized original input tensors [B, T, H]
// @param extracted     linearized extracted tensors, slice at index t [B, 1, H]
// @param B             number of batches
// @param T             sequence length (patch length + 1)
// @param H             hidden dimension size
// @param t             index t in the sequence T to be sliced
//
__global__ void slice_tensor_at_t_kernel(floatX* orig, floatX* extracted,
                                         int B, int T, int H, int t){
    
    int b_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int h_dim = blockIdx.y * blockDim.y + threadIdx.y;
    if(b_dim < B && h_dim < H){
        extracted[b_dim*H + h_dim] = orig[b_dim*T*H + t*H + h_dim];
    }
    __syncthreads();
}

// Matmul forward function with slicing at t.
//
// (B, T, H) => slice at index t => (B, 1, H) * weight(H, NC) + bias(NC) => (B, 1, NC)
//
// @param x             linearized input tensors
// @param y             linearized output tensors
// @param weight        linearized weight tensors
// @param bias          linearized bias tensors
// @param B             number of batches
// @param T             sequence length (patch length + 1)
// @param H             hidden dimension size
// @param NC            number of classes
// @param t             index t in the sequence T to be sliced 
//
__global__ void matmul_forward_with_slicing_at_t_kernel(floatX* x, floatX* y, floatX* weight, floatX* bias,
                                                        int B, int T, int H, int NC, int t){
    
    int b_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int nc_dim = blockIdx.y * blockDim.y + threadIdx.y;
    if(b_dim < B && nc_dim < NC){
        floatX val = (bias != NULL) ? bias[nc_dim] : 0.f;
        for(int h=0; h<H; ++h){
            floatX extracted_x = x[b_dim*T*H + t*H + h];
            val += extracted_x * weight[h*NC + nc_dim];
        }

        y[b_dim*NC + nc_dim] = val;
    }
}

// Matmul backward function with slicing at t.
//
// @param x             linearized input tensors [B, T, H]
// @param weight        linearized weight tensors [H, NC]
// @param dx            linearized input tensor derivatives [B, T, H]
// @param dweight       linearized weight tensor derivatives [H, NC]
// @param dbias         linearized bias tensor derivatives [NC]
// @param dy            linearized output tensor derivatives [B, 1, NC]
// @param B             number of batches
// @param T             sequence length (patch length + 1)
// @param H             hidden dimension size
// @param NC            number of classes
// @param t             index t in the sequence T to be sliced 
//
__global__ void matmul_backward_with_slicing_at_t_kernel(floatX* x, floatX* weight, floatX* dx, floatX* dweight, floatX* dbias,
                                                         floatX* dy, int B, int T, int H, int NC, int t){
    
    int b_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int nc_dim = blockIdx.y * blockDim.y + threadIdx.y;
    if(b_dim < B && nc_dim < NC){
        floatX grad = dy[b_dim*NC + nc_dim];

        if(dbias!=NULL){
            atomicAdd(&dbias[nc_dim], grad);
        }
        for(int h=0; h<H; ++h){
            floatX extracted_x = x[b_dim*T*H + t*H + h];

            atomicAdd(&dweight[h*NC + nc_dim], grad*extracted_x);
            atomicAdd(&dx[b_dim*T*H + t*H + h], grad*weight[h*NC + nc_dim]);
        }
    }
}

// -----------------------------------------------------------------------------------------
// kernel launcher

void matmul_forward1(floatX* x, floatX* y, floatX* weight, floatX* bias,
                     int B, int in_r, int in_c, int ou_c, const int sqrt_block_size){
    
    dim3 gridDim(ceil_div(B*in_r, sqrt_block_size), ceil_div(ou_c, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel1<<<gridDim, blockDim>>>(x, y, weight, bias, B*in_r, in_c, ou_c);
}

void matmul_backward1(floatX* x, floatX* weight, floatX* dx, floatX* dweight, floatX* dbias,
                      floatX* dy, int B, int in_r, int in_c, int ou_c, const int sqrt_block_size){
    
    dim3 gridDim(ceil_div(B*in_r, sqrt_block_size), ceil_div(ou_c, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_backward_kernel<<<gridDim, blockDim>>>(x, weight, dx, dweight, dbias, dy, B*in_r, in_c, ou_c);
}

void matmul_forward_with_slicing_at_t1(floatX* x, floatX* y, floatX* weight, floatX* bias,
                                       int B, int T, int H, int NC, int t, const int sqrt_block_size){
    
    floatX* extracted_x;
    cudaMalloc(&extracted_x, B*H*sizeof(floatX));

    dim3 slice_gridDim(ceil_div(B, sqrt_block_size), ceil_div(H, sqrt_block_size));
    dim3 matmul_gridDim(ceil_div(B, sqrt_block_size), ceil_div(NC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);

    slice_tensor_at_t_kernel<<<slice_gridDim, blockDim>>>(x, extracted_x, B, T, H, t);
    matmul_forward_kernel1<<<matmul_gridDim, blockDim>>>(extracted_x, y, weight, bias, B, H, NC);

    cudaDeviceSynchronoize();
    cudaFree(extracted_x);
}

void matmul_forward_with_slicing_at_t2(floatX* x, floatX* y, floatX* weight, floatX* bias,
                                       int B, int T, int H, int NC, int t, const int sqrt_block_size){
    
    dim3 gridDim(ceil_div(B, sqrt_block_size), ceil_div(NC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);

    matmul_forward_with_slicing_at_t_kernel<<<gridDim, blockDim>>>(x, y, weight, bias, B, T, H, NC, t);

}

void matmul_backward_with_slicing_at_t(floatX* x, floatX* weight, floatX* dx, floatX* dweight, floatX* dbias,
                                       floatX* dy, int B, int T, int H, int NC, int t, const int sqrt_block_size){
    
    dim3 gridDim(ceil_div(B, sqrt_block_size), ceil_div(NC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);

    matmul_backward_with_slicing_at_t_kernel<<<gridDim, blockDim>>>(x, weight, dx, dweight, dbias, dy, B, T, H, NC, t);

}

void matmul_cublaslt(const floatX* a, const floatX* b, const floatX* bias, floatX* d, int m, int n, int k,
                     cudaStream_t stream, bool transA, bool transB, int batch_count,
                     size_t strideA, size_t strideB, size_t strideD, bool accumulate,
                     floatX* pre_gelu, bool backward){
    NVTX_RANGE_FN();

    bool has_bias = (bias!=NULL);
    bool has_gelu = (pre_gelu!=NULL);

    // check alignment
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0){
        printf("[ERROR] All cuBLASLt pointers must be aligned!\n");
        exit(EXIT_FAILURE);
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    // create parameters for cuBLAS heuristic algorithm search, and preference settings
    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    // set descriptor attribute - transpose matrix
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t CLayout;
    cublasLtMatrixLayout_t DLayout;
    if(transA){
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k));
    } else{
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m));
    }
    if(transB){
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n));
    } else{
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k));
    }
    // cuBLASLt requires C in FP8 mode to be either BF16 or FP32
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, (sizeof(floatX)==1) ? CUDA_R_32F : CUBLAS_LOWP, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));

    // strided batched GEMM (used for non-flash attention, equivalent to cublasGemmStridedBatchedEx)
    if(batch_count){
        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
    }

    // create a preference handle with the specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue;
    if(has_gelu){
        int64_t gelu_ld = m;
        // setting the leading dim (colum-major) layout attribute for gelu op.
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        // setting the data pointer to the postprocessing (gelu)
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));

        if(backward){
            assert(!has_bias); // we shouldn't have any backward matmuls that use both GELU and bias.
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else{
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if(has_bias){
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else{
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if(has_bias){
        // cuBLASLt requires C in FP8 mode to be either BF16 or FP32
        cublasDataType_t bias_data_type = (sizeof(floatX)==1) ? CUDA_R_32F : CUBLAS_LOWP;
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CuBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, otherwise, even for FP8, set to FP32)
    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout, preference, 1, &heuristic, &returnedResults);
    if(returnedResults==0){
        printf("[ERROR] No cuBLASLt algorithm found, m: %d, n: %d, k: %d, bias : %d", m, n, k, has_bias);
        exit(EXIT_FAILURE);
    }

    // set whether to accumulate or not - this isn't considred in algorithm selection (?!)
    const float alpha = 1.f, beta = accumulate ? 1.f : 0.f;

    // call the matmul, finally!
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));
    
    // resource cleanups
    cublasCheck(cublasLtMatmulPreferenceDestropy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}

void matmul_forward_cublaslt(floatX* x, floatX* y, floatX* weight, floatX* bias,
                             int B, int in_r, int in_c, int ou_c, cudaStream_t stream, floatX* pre_gelu){
    
    matmul_cublaslt(weight, x, bias, y, ou_c, in_r, in_c, stream, false, false, 0, 0, 0, 0, false, NULL, false);
}