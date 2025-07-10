#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t res, const char *func, const char *file, const int line)
{
	if (res == cudaSuccess)
		return;
	std::cerr << "CUDA error = " << static_cast<unsigned int>(res);
	std::cerr << " at " << file << ":" << line << " '" << func << "' \n";
	cudaDeviceReset();
	exit(1);
}

__global__ void hello(cudaTextureObject_t tex)
{
	float idx = threadIdx.z * blockDim.y * blockDim.x
	                 + threadIdx.y * blockDim.x
	                 + threadIdx.x;
	
	idx *= 1.3;
	unsigned char val = tex1Dfetch<unsigned char>(tex, idx);

	printf("I'm %.2f and got %c\n", idx, val);
}

#define N 8

int main()
{
	dim3 blocks;
	dim3 threads(2, 2, 2);
	unsigned char h_mem[N] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};

	unsigned char *d_mem;
	CHECK(cudaMalloc(&d_mem, N * sizeof(unsigned char)));
	CHECK(cudaMemcpy(d_mem, h_mem, N * sizeof(unsigned char), cudaMemcpyHostToDevice));

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = d_mem;
	resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
	resDesc.res.linear.desc.x = 8;
	resDesc.res.linear.sizeInBytes = N * sizeof(unsigned char);

	cudaTextureDesc texDesc = {};
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t tex = 0;
	CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

	hello<<<blocks, threads>>>(tex);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaDestroyTextureObject(tex));
	CHECK(cudaFree(d_mem));
}
