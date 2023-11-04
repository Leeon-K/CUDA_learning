#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <string>


int main(){
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0){
        std::cout << "There is no device supporting CUDA" << std::endl;
    }
    else{
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)" << std::endl;
    }
    for (int dev=0; dev < deviceCount; dev++){
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        // xian cun rong liang
        printf("  Total amount of global memory:                 %.0f MBytes "
                "(%llu bytes)\n",
                static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                (unsigned long long)deviceProp.totalGlobalMem);
        // clk hz
        printf( "  GPU Max Clock rate:                            %.0f MHz (%0.2f " // 1755 MHz (1.75 GHz)
            "GHz)\n",
            deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        //if (deviceProp.l2CacheSize) {
        printf("  L2 Cache Size:                                 %d bytes\n",      // 6'291'456 bytes
                deviceProp.l2CacheSize);
        //}
        // high-frequent used
        printf("  Total amount of shared memory per block:       %zu bytes\n",     // 49'152 bytes
            deviceProp.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n",     // 102'400 bytes
            deviceProp.sharedMemPerMultiprocessor);
        printf("  Total number of registers available per block: %d\n",            // 65'536
            deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",            // 32
            deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n",            // 1536
            deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",            // 1024
            deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a block size (x,y,z): (%d, %d, %d)\n",     // (1024, 1024, 64)
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",   // (2147483647, 65535, 65535)
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);
    }
    return 0;
}