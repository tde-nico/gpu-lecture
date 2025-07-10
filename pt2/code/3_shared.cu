#include <iostream>
#include <cuda.h>

#define TILE_SIZE 16

// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
#define CHECK(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t res, const char *func, const char *file, const int line) {
	if (!res)
		return ;
	std::cerr << "CUDA error = " << static_cast<unsigned int>(res);
	std::cerr << " at " << file << ":" << line << " '" << func << "' \n";
	cudaDeviceReset();
	exit(1);
}

__global__ void matrix_mul_kernel(float *M, float *N, float *P, int side) {
	__shared__ float tile_M[TILE_SIZE][TILE_SIZE];
	__shared__ float tile_N[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	float sum = 0;

	if (row >= side || col >= side)
		return;

	for (int t = 0; t < (side + TILE_SIZE - 1) / TILE_SIZE; ++t) {
		if (t * TILE_SIZE + threadIdx.x < side)
			tile_M[threadIdx.y][threadIdx.x] = M[row * side + t * TILE_SIZE + threadIdx.x];

		if (t * TILE_SIZE + threadIdx.y < side)
			tile_N[threadIdx.y][threadIdx.x] = N[(t * TILE_SIZE + threadIdx.y) * side + col];

		__syncthreads();

		for (int k = 0; k < TILE_SIZE; ++k)
			sum += tile_M[threadIdx.y][k] * tile_N[k][threadIdx.x];

		__syncthreads();
	}

	P[row * side + col] = sum;
}

void matrix_mul(float *M, float *N, float *P, int side) {
	unsigned long long size = side * side * sizeof(float);
	float *d_M, *d_N, *d_P;
	CHECK(cudaMalloc(&d_M, size));
	CHECK(cudaMalloc(&d_N, size));
	CHECK(cudaMalloc(&d_P, size));
	CHECK(cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice));

	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 blocks((side + TILE_SIZE - 1) / TILE_SIZE, (side + TILE_SIZE - 1) / TILE_SIZE);
	matrix_mul_kernel<<<blocks, threads>>>(d_M, d_N, d_P, side);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_M));
	CHECK(cudaFree(d_N));
	CHECK(cudaFree(d_P));
}
