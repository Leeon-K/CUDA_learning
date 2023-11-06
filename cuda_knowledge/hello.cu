#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include<iostream>

// __global__  是CUDA kernel函数前缀，该函数被CPU调用启动，在GPU上执行
// blockidx.x 是 block的ID；  blockDim.x 是 block内线程数量； threadIdx.x 是线程的id 
__global__ void hello_cuda(){
    // threadIdx.x + blockDim.x * blockIdx.x;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    printf("hello cuda from {blockDim.y %d,blockIdx.x %d,blockIdx.y %d,threadIdx.x %d, threadIdx.y %d,griddim.x=%d,tx=%d,ty=%d}\n",
    blockDim.y, blockIdx.x,blockIdx.y, threadIdx.x,threadIdx.y, gridDim.x,tx,ty);
    // unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("[%d] hello cuda from {blockIdx %d,blockDim %d,threadIdx %d,griddim.x=%d}\n", 
    // idx, blockIdx.x, blockDim.x, threadIdx.x, gridDim.x);
}

int main(){
    //<<<>>> 启动CUDA kernel的标志 第一个数表示分配的block数量，第二个表示每个block中的线程数量
    dim3 gridSize(4, 4);  //gridDim.x  .y  .z
    dim3 blockDim(5, 5); //blockDim.x .y  .z
    hello_cuda<<<gridSize, blockDim>>>();  //gridSize  blockDim 都可以是多维的
    // hello_cuda<<<3,4>>>();  //     griddim.x  .y                       blockdim
    // 该函数处强制CPU等待GPU上的CUDA kernel执行，即同步，这里也可以不用写，只不过CPU会比GPU先执行完罢了
    // cudaDeviceSynchronize();
    return 0;
}