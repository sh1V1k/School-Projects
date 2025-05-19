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

#define TILE_SZ_A 64
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

/* NOTE: A and C are column major, B is row major
  */
__global__ void mygemm(float * __restrict__ c,       //<! [out] and MxN matrix
                        const float *a, //<! [in] an MxK matrix
                        const float *b, //<! [in] an KxN matrix
                        const int M, const int N, const int K) {

// Macros for accessing flattened matrices
#define A(_i, _j) a[(_i)*K + (_j)]
#define B(_i, _j) b[(_i)*N + (_j)]
#define C(_i, _j) c[(_i)*N + (_j)]

  // Shared memory for tiling input B array
  __shared__ float B_s[TILE_SZ_RATIO][TILE_SZ_B];

  // Index variables
  const unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int col = blockIdx.y * TILE_SZ_B;

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
    const unsigned int i = threadIdx.x / TILE_SZ_B;
    const unsigned int j = threadIdx.x % TILE_SZ_B;
    if (tileIdx * TILE_SZ_RATIO + i < K && col + j < N) {
      B_s[i][j] = B(tileIdx * TILE_SZ_RATIO + i, col + j);
    } else {
      B_s[i][j] = 0;
    }
    __syncthreads();
    // Loop over elements inside the tile
    for (unsigned int idx = 0; idx < TILE_SZ_RATIO; ++idx) {
      // Load tile of A matrix into register
      float a_reg;
      if (row < M && tileIdx * TILE_SZ_RATIO + idx < K) {
        a_reg = A(row, tileIdx * TILE_SZ_RATIO + idx);
      } else {
        a_reg = 0;
      }
      // Loop over and update the output elements assigned to the thread
      for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
        c_reg[outIdx] += a_reg * B_s[idx][outIdx];
      }
    }
    __syncthreads();
  }

  for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx) {
    if (row < M && col + outIdx < N) {
      C(row, col + outIdx) = c_reg[outIdx];
    }
  }

#undef A
#undef B
#undef C
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
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  size = numCRows * numCColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(size * sizeof(float));

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
  
  dim3 dimGrid((numARows + TILE_SZ_A - 1) / TILE_SZ_A, (numBColumns +TILE_SZ_B - 1) / TILE_SZ_B);
  dim3 dimBlock(TILE_SZ_A, 1);

  //@@ Launch the GPU Kernel here

  mygemm<<<dimGrid, dimBlock>>>(deviceC, deviceA, deviceB, numARows, numBColumns, numAColumns);

  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here

  wbCheck(cudaMemcpy(hostC, deviceC, size*sizeof(float), cudaMemcpyDeviceToHost));

  //@@ Free the GPU memory here

  wbCheck(cudaFree(deviceA));
  wbCheck(cudaFree(deviceB));
  wbCheck(cudaFree(deviceC));


  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix

  return 0;
}
