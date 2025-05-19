/* Stream implementation */

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define NSTREAM 4

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
    int gridW = (Width_out+TILE_WIDTH-1) / TILE_WIDTH;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(size_t)(i3) * (Channel * Height * Width) + (size_t)(i2) * (Height * Width) + (size_t)(i1) * (Width) + (size_t)i0]

    // TODO: Insert your input matrix unrolling kernel code here
    int b = blockIdx.z;
    int h = (blockIdx.y / gridW )* TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % gridW)* TILE_WIDTH + threadIdx.x;
    int h_unroll;
    int w_unroll;
    int w_base;
    if (b < Batch && h < Height_out && w < Width_out){
            for (int c = 0; c < Channel; c ++){
                w_base = c * K * K;
                for (int p = 0; p < K; p ++){
                    for (int q = 0; q < K; q ++){
                        h_unroll = w_base + p * K + q;
                        w_unroll = h * Width_out + w;
                        output[ ((size_t)h_unroll * Height_out * Width_out * Batch) + (size_t)(b *  Height_out * Width_out) + (size_t)w_unroll ] = in_4d(b,c, h + p, w + q);
                    }
                }
            }
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
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
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

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out / NSTREAM;
    const int out_image_size = Height_out * Width_out;
    cudaStream_t streams[NSTREAM];
    float *unrolled_matrix[NSTREAM];  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output[NSTREAM];    // Pointer to device memory for storing the result of matrix multiplication

    for (int i = 0; i < NSTREAM; i ++){
        cudaMalloc((void**)&unrolled_matrix[i], (size_t)Batch * Channel * K * K * Height_out * Width_out * sizeof(float) / NSTREAM);
        cudaMalloc((void**)&matmul_output[i], ((size_t)Batch * Map_out * Height_out * Width_out) * sizeof(float) / NSTREAM);
        cudaStreamCreate(&streams[i]);
    }

    cudaHostRegister((void*)host_input, (size_t)Batch * Channel * Height * Width * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister((void*)host_mask, K * K * Channel * Map_out  * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister((void*)host_output, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float), cudaHostRegisterDefault);

    cudaMalloc((void **)device_output_ptr, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, (size_t)Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, K * K * Channel * Map_out * sizeof(float));

    cudaMemcpy(*device_mask_ptr, host_mask, K * K * Channel * Map_out * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 gridDim(1, ceil(1.0 * (Width_out ) / TILE_WIDTH) * ceil(1.0 * Height_out /  TILE_WIDTH),  Batch / NSTREAM );
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    dim3 gridDim1(ceil(1.0 * Width_unrolled / TILE_WIDTH), ceil( 1.0 * Map_out / TILE_WIDTH), 1);
    dim3 blockDim1(TILE_WIDTH, TILE_WIDTH,1);

    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch / NSTREAM, 1);

    
    for (size_t i = 0; i < NSTREAM; i++){

        size_t offset_in = (size_t)Batch * Channel * Height * Width * i/ NSTREAM;
        size_t offset_unroll = (size_t)Channel * Height * Width * Batch * i / NSTREAM;
        int offset_permute = Batch * Map_out * out_image_size * i / NSTREAM;
        size_t offset_out = (size_t)Batch * Map_out * (Height - K + 1) * (Width - K + 1) * i/ NSTREAM;
        cudaMemcpyAsync(*device_input_ptr + offset_in, host_input + offset_in, (size_t)Batch * Channel * Height * Width * sizeof(float) / NSTREAM, cudaMemcpyHostToDevice, streams[i]);
        matrix_unrolling_kernel<<<gridDim, blockDim, 0, streams[i]>>>( *device_input_ptr + offset_unroll,unrolled_matrix[i],Batch / NSTREAM,Channel, Height, Width, K);
        matrixMultiplyShared<<<gridDim1, blockDim1,0 , streams[i]>>>(*device_mask_ptr, unrolled_matrix[i], matmul_output[i], Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out , Width_unrolled);
        matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE, 0, streams[i]>>>(
            matmul_output[i], *device_output_ptr + offset_permute, Map_out, Batch/ NSTREAM, out_image_size
        );
        cudaMemcpyAsync((float *)host_output + offset_out, *device_output_ptr + offset_out, (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float) / NSTREAM, cudaMemcpyDeviceToHost, streams[i]);

    }

    for (int i = 0; i< NSTREAM; i++){
        cudaFreeAsync(matmul_output[i],streams[i]);
        cudaFreeAsync(unrolled_matrix[i], streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaHostUnregister((void*)host_input);
    cudaHostUnregister((void*)host_mask);
    cudaHostUnregister((void*)host_output);

    ((cudaFree(device_output_ptr)));
    ((cudaFree(device_mask_ptr)));
    ((cudaFree(device_input_ptr)));

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

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
