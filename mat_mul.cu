#include <iostream>
#include <cuda_runtime.h>

// Error handling MACRO
#define CUDA_CHECK(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", \
      cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(err); \
  } \
} while (0)

__global__ void matrixMulKernel(float *A, float *B, float *C, int M, int K, int N) {
    // Compute row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < M && col < N) {
        float sum = 0.0f;
        // Compute the inner product
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
  // Matrix dimensions
  int M = 8192, K = 8192, N = 8192;
  size_t sizeA = (size_t)M * K * sizeof(float);
  size_t sizeB = (size_t)K * N * sizeof(float);
  size_t sizeC = (size_t)M * N * sizeof(float);

  // Allocate host memory
  float *h_A = (float *)malloc(sizeA);
  float *h_B = (float *)malloc(sizeB);
  float *h_C = (float *)malloc(sizeC);

  // Initialize host matrices
  for (size_t i = 0; i < (size_t)M * K; i++) {
    h_A[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
  }
  for (size_t i = 0; i < (size_t)K * N; i++) {
    h_B[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
  }
  memset(h_C, 0, sizeC);

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, sizeA));
  CUDA_CHECK(cudaMalloc(&d_B, sizeB));
  CUDA_CHECK(cudaMalloc(&d_C, sizeC));

  // Copy host matrices to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice ));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice ));
  CUDA_CHECK(cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice ));


  // Test 8x8, 16x16, 32x32
  int dims[] = {8, 16, 32};
  for (int d : dims) {
    dim3 threadsPerBlock(d, d);
    dim3 blocksPerGrid((N + d - 1) / d, (M + d - 1) / d);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double flops = (2.0 * M * N * K) / (ms / 1000.0);
    printf("Block %dx%d | Time: %.2f ms | TFLOPS: %.2f\n", d, d, ms, flops / 1e12);
  }

  return 0;
}
