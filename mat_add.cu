#include <iostream>
#include <cuda_runtime.h>

#define N 8192 // Matrix size N x N

// Error handling MACRO
#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", \
      cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(err); \
  } \
} while (0)

// CUDA kernel for matrix addition - 1D
__global__ void matrixAdd1D(const float* A, const float* B, float* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global thread index
  if (i < n * n) {
    C[i] = A[i] + B[i];  // Perform matrix addition
  }
}

// CUDA kernel for matrix addition - 2D
__global__ void matrixAdd2D(const float* A, const float* B, float* C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    int idx = row * n + col;
    C[idx] = A[idx] + B[idx];  // Perform matrix addition
  }
}

int main() {
  long long numElements = (long long)N * N;
  size_t size = numElements * sizeof(float);

  // Memory allocation
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  // Initialize host matrices
  for (long long i = 0; i < numElements; i++) {
    h_A[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
    h_B[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
    h_C[i] = 0.0f;
  }

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, size));
  CUDA_CHECK(cudaMalloc(&d_B, size));
  CUDA_CHECK(cudaMalloc(&d_C, size));

  // Copy host matrices to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

  // Timing setup
  cudaEvent_t start, stop;
  float milliseconds = 0;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // 1D Kernel execution
  int threadsPerBlock1D = 256;
  int blocksPerGrid1D = (numElements + threadsPerBlock1D - 1) / threadsPerBlock1D;

  CUDA_CHECK(cudaEventRecord(start));
  matrixAdd1D<<<blocksPerGrid1D, threadsPerBlock1D>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  // Compute time and FLOPs
  double flops1D = 2.0 * numElements / (milliseconds / 1000.0);
  std::cout << "1D Kernel Time: " << milliseconds << " ms, FLOPs: " << flops1D << std::endl;

  // 2D Kernel execution
  dim3 threadsPerBlock2D(32, 32);
  dim3 blocksPerGrid2D((N + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
                       (N + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);

  // Reset timing
  milliseconds = 0;
  CUDA_CHECK(cudaEventRecord(start));
  matrixAdd2D<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  // Compute time and FLOPs
  double flops2D = 2.0 * numElements / (milliseconds / 1000.0);
  std::cout << "2D Kernel Time: " << milliseconds << " ms, FLOPs: " << flops2D << std::endl;

  // Clean up
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
