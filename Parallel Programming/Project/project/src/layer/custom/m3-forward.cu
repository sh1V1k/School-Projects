#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void matrixMultiplyAndPermute(const float *A, const float *B, float *output,
    int Map_out, int Batch, int Height, int Width, int K, int Channel) {
// Shared memory for tiled multiplication
__shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
__shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
const int Height_out = Height - K + 1;
const int Width_out = Width - K + 1;
const int Height_unrolled = Channel * K * K;
const size_t Width_unrolled = Batch * Height_out * Width_out;

#define in_4d(i3, i2, i1, i0) B[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

// Grid and thread indices
int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

// Compute row and column in the C matrix
int row = by * TILE_WIDTH + ty;
int col = bx * TILE_WIDTH + tx;

float val = 0;

// Perform tiled matrix multiplication
for (int tileId = 0; tileId < (Height_unrolled - 1) / TILE_WIDTH + 1; ++tileId) {
// Load tile from A and B into shared memory
if (row < Map_out && tileId * TILE_WIDTH + tx < Height_unrolled) {
tileA[ty][tx] = A[row * Height_unrolled + tileId * TILE_WIDTH + tx];
} else {
tileA[ty][tx] = 0.0f;
}
if (col < Width_unrolled && tileId * TILE_WIDTH + ty < Height_unrolled) {
    int b = col / (Height_out * Width_out);    // Batch index
    int pixel_idx = col % (Height_out * Width_out);  // Pixel index in image
    int h_out = pixel_idx / Width_out;   // Height index in the output image
    int w_out = pixel_idx % Width_out;  // Width index in the output image
    int s = tileId * TILE_WIDTH + ty;  // This corresponds to the unrolled index

    // Unrolling the index for B as done in the original kernel
    int c = s / (K * K);
    int p = (s % (K * K)) / K;
    int q = (s % (K * K)) % K;

    tileB[ty][tx] = in_4d(b, c, h_out + p, w_out + q);
} else {
tileB[ty][tx] = 0.0f;
}
__syncthreads();

// Compute partial result
for (int i = 0; i < TILE_WIDTH; ++i) {
val += tileA[ty][i] * tileB[i][tx];
}
__syncthreads();
}

// Compute the permuted index and write the result
if (row < Map_out && col < Width_unrolled) {
int b = col / (Height_out * Width_out);       // Batch index
int pixel_idx = col % (Height_out * Width_out); // Pixel index in image
int h_out = pixel_idx / Width_out;           // Height index in the output image
int w_out = pixel_idx % Width_out;           // Width index in the output image

// Compute the permuted index
int permuted_idx = b * Map_out * Height_out * Width_out +
row * Height_out * Width_out +
h_out * Width_out + w_out;

output[permuted_idx] = val;
#undef in_4d
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
    cudaMalloc((void**)device_mask_ptr, Channel * Map_out * K * K * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Channel * Map_out * K * K * sizeof(float), cudaMemcpyHostToDevice);


}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const size_t Width_unrolled = Batch * Height_out * Width_out;

    dim3 dimGrid((Width_unrolled + TILE_WIDTH - 1) / TILE_WIDTH, (Map_out + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    matrixMultiplyAndPermute<<<dimGrid, dimBlock>>>(device_mask, device_input, device_output, Map_out, Batch, Height, Width, K, Channel);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    ((cudaMemcpy(host_output, device_output, (size_t)(Batch * Map_out * (Height-K+1)*(Width-K+1) * sizeof(float)), cudaMemcpyDeviceToHost)));

    // TODO: Free device memory

    ((cudaFree(device_output)));
    ((cudaFree(device_input)));
    ((cudaFree(device_mask)));
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