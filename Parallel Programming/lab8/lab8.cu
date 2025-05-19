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

__global__ void spmvJDSKernel(float *out, int *matColStart, int *matCols,
                              int *matRowPerm, int *matRows,
                              float *matData, float *vec, int dim) {
  //@@ insert spmv kernel for jds format
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < dim){
    float dot = 0.0;
    unsigned int sec = 0;
    while(sec < matRows[row]){ //matColStart[sec+1] - matColStart[sec] > row; is what it used to be, but we use matRows which tells us how many times a thread needs to do work
      dot += matData[matColStart[sec]+row] * vec[matCols[matColStart[sec]+row]];
      sec++;
    }
    out[matRowPerm[row]] = dot;
  }
}

static void spmvJDS(float *out, int *matColStart, int *matCols,
                    int *matRowPerm, int *matRows, float *matData,
                    float *vec, int dim) {

  //@@ invoke spmv kernel for jds format
  spmvJDSKernel<<<(dim+1023)/1024, 1024>>>(out, matColStart, matCols, matRowPerm, matRows, matData, vec, dim);
}

int main(int argc, char **argv) {
  wbArg_t args;
  int *hostCSRCols;
  int *hostCSRRows;
  float *hostCSRData;
  int *hostJDSColStart;
  int *hostJDSCols;
  int *hostJDSRowPerm;
  int *hostJDSRows;
  float *hostJDSData;
  float *hostVector;
  float *hostOutput;
  int *deviceJDSColStart;
  int *deviceJDSCols;
  int *deviceJDSRowPerm;
  int *deviceJDSRows;
  float *deviceJDSData;
  float *deviceVector;
  float *deviceOutput;
  int dim, ncols, nrows, ndata;
  int maxRowNNZ;

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  hostCSRCols = (int *)wbImport(wbArg_getInputFile(args, 0), &ncols, "Integer");
  hostCSRRows = (int *)wbImport(wbArg_getInputFile(args, 1), &nrows, "Integer");
  hostCSRData = (float *)wbImport(wbArg_getInputFile(args, 2), &ndata, "Real");
  hostVector = (float *)wbImport(wbArg_getInputFile(args, 3), &dim, "Real");

  hostOutput = (float *)malloc(sizeof(float) * dim);



  CSRToJDS(dim, hostCSRRows, hostCSRCols, hostCSRData, &hostJDSRowPerm, &hostJDSRows,
           &hostJDSColStart, &hostJDSCols, &hostJDSData);
  maxRowNNZ = hostJDSRows[0];

  // Allocate GPU memory.
  cudaMalloc((void **)&deviceJDSColStart, sizeof(int) * maxRowNNZ);
  cudaMalloc((void **)&deviceJDSCols, sizeof(int) * ndata);
  cudaMalloc((void **)&deviceJDSRowPerm, sizeof(int) * dim);
  cudaMalloc((void **)&deviceJDSRows, sizeof(int) * dim);
  cudaMalloc((void **)&deviceJDSData, sizeof(float) * ndata);

  cudaMalloc((void **)&deviceVector, sizeof(float) * dim);
  cudaMalloc((void **)&deviceOutput, sizeof(float) * dim);

  // int i;
  // printf("hostJDSColStart \n");
  // for(i = 0; i < maxRowNNZ; i++){
  //   printf("%d ", hostJDSColStart[i]);
  // }
  // printf("\n");
  // printf("hostJDSCols \n");
  // for(i = 0; i < ndata; i++){
  //   printf("%d ", hostJDSCols[i]);
  // }
  // printf("\n");
  // printf("hostJDSRowPerm \n");
  // for(i = 0; i < dim; i++){
  //   printf("%d ", hostJDSRowPerm[i]);
  // }
  // printf("\n");
  // printf("hostJDSRows \n");
  // for(i = 0; i < dim; i++){
  //   printf("%d ", hostJDSRows[i]);
  // }
  // printf("\n");
  // printf("hostJDSData \n");
  // for(i = 0; i < ndata; i++){
  //   printf("%f ", hostJDSData[i]);
  // }
  // printf("\n");
  // printf("hostVector \n");
  // for(i = 0; i < dim; i++){
  //   printf("%f ", hostVector[i]);
  // }
  // printf("\n");
  // Copy input memory to the GPU.
  cudaMemcpy(deviceJDSColStart, hostJDSColStart, sizeof(int) * maxRowNNZ,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSCols, hostJDSCols, sizeof(int) * ndata, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSRowPerm, hostJDSRowPerm, sizeof(int) * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSRows, hostJDSRows, sizeof(int) * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSData, hostJDSData, sizeof(float) * ndata, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceVector, hostVector, sizeof(float) * dim, cudaMemcpyHostToDevice);


  // Perform CUDA computation
  spmvJDS(deviceOutput, deviceJDSColStart, deviceJDSCols, deviceJDSRowPerm, deviceJDSRows,
          deviceJDSData, deviceVector, dim);
  cudaDeviceSynchronize();

  // Copy output memory to the CPU
  cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * dim, cudaMemcpyDeviceToHost);

  // Free GPU Memory
  cudaFree(deviceVector);
  cudaFree(deviceOutput);
  cudaFree(deviceJDSColStart);
  cudaFree(deviceJDSCols);
  cudaFree(deviceJDSRowPerm);
  cudaFree(deviceJDSRows);
  cudaFree(deviceJDSData);


  wbSolution(args, hostOutput, dim);

  free(hostCSRCols);
  free(hostCSRRows);
  free(hostCSRData);
  free(hostVector);
  free(hostOutput);
  free(hostJDSColStart);
  free(hostJDSCols);
  free(hostJDSRowPerm);
  free(hostJDSRows);
  free(hostJDSData);

  return 0;
}
