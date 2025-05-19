#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__constant__ float deviceMask[3150]; //3150 is an upperbound on how much we'll need. Our kernel should handle boundary checks we won't need to worry about being precise

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Image_size = Height_out * Width_out;
    const size_t Width_unrolled = Batch * Height_out * Width_out;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    // TODO: Insert your input matrix unrolling kernel code here

    size_t c, s, h_out, w_out, h_unroll, w_base, p, q, w_unroll, b;
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    if(t< Height_unrolled * Width_unrolled){
        h_unroll = t % Width_unrolled;
        w_unroll = t / Width_unrolled;
        c = w_unroll / (K*K);
        s = w_unroll % (K * K);
        b = h_unroll/(Image_size);
        w_base = h_unroll % (Image_size);
        h_out = w_base / Width_out;
        w_out = w_base % Width_out;
        p = s / K;
        q = s % K;
        output[(size_t)w_unroll * Width_unrolled + h_unroll] = in_4d(b, c, h_out + p, w_out + q); 
    } 

    #undef in_4d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = deviceMask[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    // (cudaMalloc((void**)device_output_ptr, (size_t)(Batch * Map_out * (Height-K+1)*(Width-K+1) * sizeof(float))));
    // (cudaMalloc((void**)device_input_ptr, (size_t)(Batch * Channel * Height * Width * sizeof(float))));
    // (cudaMalloc((void**)device_mask_ptr, (size_t)(Channel * Map_out * K * K * sizeof(float))));
    // (cudaMemcpy(*device_input_ptr, host_input, (size_t)(Batch * Channel * Height * Width * sizeof(float)), cudaMemcpyHostToDevice));
    // (cudaMemcpy(*device_mask_ptr, host_mask, (size_t)(Channel * Map_out * K * K * sizeof(float)), cudaMemcpyHostToDevice));

    cudaMalloc((void**)device_output_ptr, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    //cudaMalloc((void**)device_mask_ptr, Channel * Map_out * K * K * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceMask, host_mask, Channel * Map_out * K * K * sizeof(float));


}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const size_t Width_unrolled = Batch * Height_out * Width_out;

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    (cudaMalloc((void**)&unrolled_matrix, (size_t)Batch * Channel * K * K * Height_out * Width_out * sizeof(float)));
    (cudaMalloc((void**)&matmul_output, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float)));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    size_t num_threads = Height_unrolled * Width_unrolled;
    matrix_unrolling_kernel<<<((num_threads+1023)/1024), 1024>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);

    // TODO: Set the kernel dimensions and call the matmul kernel
    dim3 dimGrid1((Width_unrolled+TILE_WIDTH-1)/TILE_WIDTH, (Map_out+TILE_WIDTH-1)/TILE_WIDTH, 1);
    dim3 dimBlock1(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMultiplyShared<<<dimGrid1,dimBlock1>>>(device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);


    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );
    (cudaFree(matmul_output));
    (cudaFree(unrolled_matrix));
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    ((cudaMemcpy(host_output, device_output, (size_t)(Batch * Map_out * (Height-K+1)*(Width-K+1) * sizeof(float)), cudaMemcpyDeviceToHost)));

    // TODO: Free device memory

    ((cudaFree(device_output)));
    ((cudaFree(device_input)));
    //((cudaFree(device_mask)));
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
