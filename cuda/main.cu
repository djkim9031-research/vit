#include "activations.cuh"

int main(){

    float x[5] = {0.0, 0.1, 0.2, 0.3, 0.4};
    float y[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    float* device_x;
    float* device_y;

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&device_x, 5*sizeof(float));
    if(err!=cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc(&device_y, 5*sizeof(float));
    if(err!=cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(device_x);
        return -1;
    }

    // Copy data from host to device
    err = cudaMemcpy(device_x, x, 5*sizeof(float), cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(device_x);
        cudaFree(device_y);
        return -1;
    }

    // Launch kernel
    gelu_forward1(device_x, device_y, 5, 32);

    // Copy result from device to host
    err = cudaMemcpy(y, device_y, 5*sizeof(float), cudaMemcpyDeviceToHost);
    if(err!=cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(device_x);
        cudaFree(device_y);
        return -1;
    }

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);

    // Print results
    for(int i=0; i<5;++i){
        printf("result: %f\n", y[i]);
    }

    return 0;
}