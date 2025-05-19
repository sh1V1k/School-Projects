#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

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

#define TILE_SZ_A 32
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

/* NOTE: A and C are row major, B is row major
  */
__global__ void mygemm(float * __restrict__ c,       //<! [out] and MxN matrix
                        const float *a, //<! [in] an MxK matrix
                        const float *b, //<! [in] an KxN matrix
                        const int M, const int N, const int K) {

  // Shared memory for tiling input B array
  __shared__ float B_s[TILE_SZ_RATIO][TILE_SZ_B];

  // Index variables
  const size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t col = blockIdx.x * TILE_SZ_B;

  // Privatization of output variables
  float c_reg[TILE_SZ_B];

  // Initialize output values
  for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
    c_reg[outIdx] = 0;
  }

  // Loop over the input tiles
  for (unsigned int tileIdx = 0; tileIdx < (K - 1) / TILE_SZ_RATIO + 1;
        ++tileIdx) {
    // Load the tile of B into shared memory
    const unsigned int i = threadIdx.y / TILE_SZ_B;
    const unsigned int j = threadIdx.y % TILE_SZ_B;
    if (tileIdx * TILE_SZ_RATIO + i < K && col + j < N) {

      B_s[i][j] = b[(tileIdx * TILE_SZ_RATIO + i) * N + (col + j)];
    } else {
      B_s[i][j] = 0.0f;
    }
    __syncthreads();
    // Loop over elements inside the tile
    for (unsigned int idx = 0; idx < TILE_SZ_RATIO; ++idx) {
      // Load tile of A matrix into register
      float a_reg;
      if (row < M && tileIdx * TILE_SZ_RATIO + idx < K) {
        a_reg = a[row * K + (tileIdx * TILE_SZ_RATIO + idx)];
      } else {
        a_reg = 0.0f;
      }
      // Loop over and update the output elements assigned to the thread
      for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
        c_reg[outIdx] += a_reg * B_s[idx][outIdx];
      }
    }
  }

    __syncthreads();

  for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
    if (row < M && col + outIdx < N) {
      c[row * N + (col + outIdx)] = c_reg[outIdx];
    }
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
    cudaMalloc((void**)device_mask_ptr, Channel * Map_out * K * K * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Channel * Map_out * K * K * sizeof(float), cudaMemcpyHostToDevice);

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
  //dim3 dimGrid((Map_out + TILE_SZ_A - 1) / TILE_SZ_A, (Width_unrolled +TILE_SZ_B - 1) / TILE_SZ_B);

    dim3 dimGrid((Width_unrolled +TILE_SZ_B - 1) / TILE_SZ_B, (Map_out + TILE_SZ_A - 1) / TILE_SZ_A);
    //dim3 dimBlock(TILE_SZ_A, 1);
    dim3 dimBlock(1, TILE_SZ_A);
    
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    mygemm<<<dimGrid, dimBlock>>>(matmul_output, device_mask, unrolled_matrix, Map_out, Width_unrolled, Height_unrolled);
    //matrixMultiplyShared<<<dimGrid1,dimBlock1>>>(device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }


    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    (cudaFree(matmul_output));
    (cudaFree(unrolled_matrix));
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    ((cudaMemcpy(host_output, device_output, (size_t)(Batch * Map_out * (Height-K+1)*(Width-K+1) * sizeof(float)), cudaMemcpyDeviceToHost)));

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

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
