// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>
#include<iostream>


#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  int tx = threadIdx.x; int bdx = blockDim.x;
  __shared__ float XY[2*BLOCK_SIZE];
  int i = blockIdx.x * bdx * 2 + tx;
  if(i < len){ XY[tx] = input[i]; }
  if(i + bdx < len){ XY[tx+bdx] = input[i+bdx]; }

  for(int stride = 1; stride < 2*BLOCK_SIZE; stride *= 2){
    __syncthreads();
    int index = (threadIdx.x+1) * 2* stride -1;
    if(index < 2*BLOCK_SIZE && index-stride >= 0){ 
      XY[index] += XY[index - stride];
    }
  }

  for(int stride = BLOCK_SIZE/2; stride > 0; stride /= 2){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index + stride < 2*BLOCK_SIZE){
      XY[index + stride] += XY[index];
    }
  }

  __syncthreads();
  if(i < len){ 
  output[i] = XY[tx]; 
  //printf("Inside scan kernel value at index %d = %f\n", i, output[i]);
}
  if(i + bdx < len){ 
    output[i+bdx] = XY[tx+bdx];
  //printf("Inside scan kernel value at index %d = %f\n", i+blockDim.x, output[i+blockDim.x]);
  }

}

__global__ void scan_final(float *output, float *scan_sums, int len) {
	int bx = blockIdx.x;
	int index = bx * blockDim.x + threadIdx.x;
  //printf("Inside scan_final kernel value at index %d = %f\n", index, output[index]);
	if(index < len && bx > 0){output[index] += scan_sums[bx-1]; }
}
int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *AuxArray;
  float *AuxDevice;
  float *AuxSum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  AuxArray = (float *)malloc(1024 * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&AuxSum, 1024 * sizeof(float)));
  wbCheck(cudaMalloc((void **)&AuxDevice, 1024 * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  //wbLog(TRACE, "The number of elements is ", numElements);
  //wbLog(TRACE, "The size of auxillary array is ", (numElements-1)/(2*BLOCK_SIZE) + 1);
  dim3 dimGrid((numElements-1)/(2*BLOCK_SIZE) + 1, 1, 1); //we do this since each thread loads two elements
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);

  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  //Store Block Sum to Auxillary Array
  int i = 2*BLOCK_SIZE-1;
  for(unsigned int j = 0; j < (numElements-1)/(2*BLOCK_SIZE) + 1; j++){
    AuxArray[j] = hostOutput[i];
    //printf("AuxArray value at index %d = %f\n", j, AuxArray[j]);
    i += 2*BLOCK_SIZE;
  }

  wbCheck(cudaMemcpy(AuxDevice, AuxArray, 1024 * sizeof(float),
                     cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  dim3 dimGrid1(1,1,1);
  dim3 dimBlock1((numElements-1)/(2*BLOCK_SIZE) + 1,1,1);
  scan<<<dimGrid1, dimBlock1>>>(AuxDevice, AuxSum, (numElements-1)/(2*BLOCK_SIZE) + 1);

  cudaDeviceSynchronize();
  scan_final<<<dimGrid, 1024>>>(deviceOutput, AuxSum, numElements);
  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

//  for(int i = 0; i < numElements; i++){
//    printf("hostData at index %d = %f\n", i, hostOutput[i]);
//  }

  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(AuxDevice);
  cudaFree(AuxSum);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

