#include <iostream>
#include <cuda.h>

// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
#define CHECK(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t res, const char *func, const char *file, const int line)
{
	if (!res)
		return ;
	std::cerr << "CUDA error = " << static_cast<unsigned int>(res);
	std::cerr << " at " << file << ":" << line << " '" << func << "' \n";
	cudaDeviceReset();
	exit(1);
}

__constant__ unsigned char mem[8];

__global__ void	hello()
{
	unsigned int idx = threadIdx.z * blockDim.y * blockDim.x
					 + threadIdx.y * blockDim.x
					 + threadIdx.x;

	printf("I'm %u and got %c\n", idx, mem[idx]);
}

int main(int argc, char** argv)
{
	dim3	blocks;
	dim3	threads(2, 2, 2);

	int offset = 0;
	CHECK(cudaMemcpyToSymbol(mem, "abcdefgh", 8 * sizeof(unsigned char), offset, cudaMemcpyHostToDevice));

	hello<<<blocks, threads>>>();
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
}
