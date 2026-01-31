#include <iostream>
#include <cuda_runtime.h>
#include <vector>

# error handling MACRO
#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", \
      cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(err); \
  } \
} while (0)

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global thread index
  if (i < n) {
    C[i] = A[i] + B[i];  // Perform vector addition
  }
}

int main() {
  // Vector length 2^30
  long long n = 1LL << 30;
  size_t size = n * sizeof(float);

  // Allocate host memory
  float *h_A = (float)malloc(size);
  float *h_B = (float)malloc(size);
  float *h_C = (float)malloc(size);

  // Initialize host vectors with random values 0.0 to 100.0
  // To ensure robustness, we use a simple random initialization
  for (int i=0; i < n; i++) {
    h_A[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
    h_B[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
  }

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, size));
  CUDA_CHECK(cudaMalloc(&d_B, size));
  CUDA_CHECK(cudaMalloc(&d_C, size));

  // Copy host vectors to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  // Loop through different block sizes
  int blockSizes[] = {32, 64, 128, 256};

  for (int threadsPerBlock : blockSizes) {
    // Calculate grid size
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Timing setup
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Calculate FLOPs
    double seconds = milliseconds / 1000.0;
    double flops = n / seconds;
    double gflops = flops / 1e9;
    std::cout << "Block Size: " << threadsPerBlock 
              << ", Time: " << milliseconds << " ms"
              << ", GFLOPS: " << gflops << std::endl;
  }

  // CLean up
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);

  return 0; 
}
