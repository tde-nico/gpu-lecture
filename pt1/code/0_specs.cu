#include <iostream>
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

void print_gpu_specs(int device_id)
{
	cudaDeviceProp prop;
	CHECK(cudaGetDeviceProperties(&prop, device_id));

	std::cout << "Device " << device_id << ": " << prop.name << "\n";
	std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
	std::cout << "  Total global memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
	std::cout << "  Shared memory per block: " << (prop.sharedMemPerBlock >> 10) << " KB\n";
	std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
	std::cout << "  Warp size: " << prop.warpSize << "\n";
	std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
	std::cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
	std::cout << "  Number of SMs: " << prop.multiProcessorCount << "\n";
	std::cout << "  Clock rate: " << (prop.clockRate / 1000) << " MHz\n";
	std::cout << "  Memory clock rate: " << (prop.memoryClockRate / 1000) << " MHz\n";
	std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits\n";
	std::cout << "  Max grid size: [" 
			  << prop.maxGridSize[0] << ", "
			  << prop.maxGridSize[1] << ", "
			  << prop.maxGridSize[2] << "]\n";
	std::cout << "  Max threads dim (per block): [" 
			  << prop.maxThreadsDim[0] << ", "
			  << prop.maxThreadsDim[1] << ", "
			  << prop.maxThreadsDim[2] << "]\n";
	std::cout << "  L2 cache size: " << (prop.l2CacheSize >> 10) << " KB\n";
	std::cout << "  Compute mode: " << prop.computeMode << "\n";
	std::cout << "  Concurrent kernels: " << (prop.concurrentKernels ? "Yes" : "No") << "\n";
	std::cout << "  Unified addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << "\n";
	std::cout << "  ECC enabled: " << (prop.ECCEnabled ? "Yes" : "No") << "\n";
	std::cout << std::endl;
}

int main()
{
	int device_count = 0;
	CHECK(cudaGetDeviceCount(&device_count));

	if (device_count == 0)
	{
		std::cerr << "No CUDA devices found.\n";
		return 1;
	}

	std::cout << "Found " << device_count << " CUDA device(s).\n\n";

	for (int i = 0; i < device_count; ++i) {
		print_gpu_specs(i);
	}

	return 0;
}
