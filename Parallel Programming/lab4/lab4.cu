#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 4
#define MASK_WIDTH 3
//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH*MASK_WIDTH*MASK_WIDTH]; //convolution kernel

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  __shared__ float tile[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];

  if(i < x_size && j < y_size && k < z_size) {
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = input[(k * y_size * x_size) + (j * x_size) + i];
} else {
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
}

  __syncthreads();

  int radius = MASK_WIDTH/2;
  int xs = blockIdx.x * blockDim.x;
  int ys = blockIdx.y * blockDim.y;
  int zs = blockIdx.z * blockDim.z;
  
  int xn = (blockIdx.x+1) * blockDim.x;
  int yn = (blockIdx.y+1) * blockDim.y;
  int zn = (blockIdx.z+1) * blockDim.z;

  int txs = i - radius;
  int tys = j - radius; 
  int tzs = k - radius;

  float Output = 0;
  for(int x = 0; x < MASK_WIDTH; x++){
    for(int y = 0; y < MASK_WIDTH; y++){
      for(int z = 0; z < MASK_WIDTH; z++){
        int xdx = txs + x;
        int ydx = tys + y;
        int zdx = tzs + z;

        if(xdx >= 0 && xdx < x_size && ydx >= 0 && ydx < y_size && zdx >= 0 && zdx < z_size) {
          if(xdx >= xs && xdx < xn && ydx >= ys && ydx < yn && zdx >= zs && zdx < zn){
            Output += tile[threadIdx.z-radius+z][threadIdx.y-radius+y][threadIdx.x-radius+x] * deviceKernel[(z * MASK_WIDTH * MASK_WIDTH) + (y * MASK_WIDTH) + x];
        } else {
            Output += input[(zdx * y_size * x_size) + (ydx * x_size) + xdx] * deviceKernel[(z * MASK_WIDTH * MASK_WIDTH) + (y * MASK_WIDTH) + x];
        }
        }

      }
    } 
  }
  if(i < x_size && j < y_size && k < z_size) {output[(k * y_size * x_size) + (j * x_size) + i] = Output;}
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  float *deviceInput;
  float *deviceOutput;

  wbCheck(cudaMalloc((void**)&deviceInput, (inputLength - 3)*sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceOutput, (inputLength - 3)*sizeof(float)));  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput, &hostInput[3],(inputLength - 3)*sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength*sizeof(float)));

  //@@ Initialize grid and block dimensions here
  dim3 dimGrid((x_size+TILE_WIDTH-1)/TILE_WIDTH,(y_size+TILE_WIDTH-1)/TILE_WIDTH,(z_size+TILE_WIDTH-1)/TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,TILE_WIDTH);
  //@@ Launch the GPU kernel here

  conv3d<<<dimGrid,dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)

  wbCheck(cudaMemcpy(&hostOutput[3], deviceOutput,(inputLength - 3)*sizeof(float), cudaMemcpyDeviceToHost));

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory

  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

