// LAB 2 FA24

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here

  int Row = blockIdx.y*blockDim.y+threadIdx.y;
  int Col = blockIdx.x*blockDim.x+threadIdx.x;

  if((Row < numCRows) && (Col < numCColumns)){
    float Cvalue = 0.0;

    for(int k = 0; k < numAColumns; k++){ //numAColumns could be numBRows since they should be equivalent
      Cvalue += A[(Row*numAColumns)+k] * B[(k*numBColumns)+Col];
    }
    C[Row*numCColumns+Col] = Cvalue;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  int size;

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Set numCRows and numCColumns

  numCRows = numARows;
  numCColumns = numBColumns;
  size = numCRows*numCColumns;

  //@@ Allocate the hostC matrix

  hostC = (float *)malloc(size * sizeof(float));

  if(numAColumns == numBRows){ //ensure we can do matrix multiplication

    //@@ Allocate GPU memory here

    float *deviceA;
    float *deviceB;
    float *deviceC;

    wbCheck(cudaMalloc((void**)&deviceA , numARows*numAColumns*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceB , numBRows*numBColumns*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceC , size*sizeof(float)));

    //@@ Copy memory to the GPU here

    wbCheck(cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice));


    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid((numCColumns+31)/32, (numCRows+31)/32, 1);
    dim3 dimBlock(32,32,1);

    //@@ Launch the GPU Kernel here

    matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaDeviceSynchronize();
    
    //@@ Copy the GPU memory back to the CPU here

    wbCheck(cudaMemcpy(hostC, deviceC, size*sizeof(float), cudaMemcpyDeviceToHost));

    //@@ Free the GPU memory here

    wbCheck(cudaFree(deviceA));
    wbCheck(cudaFree(deviceB));
    wbCheck(cudaFree(deviceC));
  }
  

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  //@@Free the hostC matrix
  free(hostC);

  return 0;
}

