#include <iostream>
#include <cuda.h>

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
	int i = threadIdx.x;
	int j = threadIdx.y;
	float sum = 0;
	for (int k = 0; k < side; ++k) {
		float m = M[i * side + k];
		float n = N[k * side + j];
		sum += m * n;
	}
	P[i * side + j] = sum;
}

void matrix_mul(float *M, float *N, float *P, int side) {
	unsigned long long size = side * side * sizeof(float);
	float *d_M, *d_N, *d_P;
	CHECK(cudaMalloc(&d_M, size));
	CHECK(cudaMalloc(&d_N, size));
	CHECK(cudaMalloc(&d_P, size));
	CHECK(cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice));

	dim3 blocks(1, 1, 1);
	dim3 threads(side, side, 1);
	matrix_mul_kernel<<<blocks, threads>>>(d_M, d_N, d_P, side);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	
	CHECK(cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost));
	
	CHECK(cudaFree(d_M));
	CHECK(cudaFree(d_N));
	CHECK(cudaFree(d_P));
}
