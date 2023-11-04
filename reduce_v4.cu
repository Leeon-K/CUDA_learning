#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"

// latency = 0.121856 ms
__device__ void WarpSharedMemReduce(volatile float* smem, int tid){
    float x = smem[tid]; // 读取全局内存到shared mem，每个线程有独立的调用栈, 不同步
    if (blockDim.x >= 64) {
      x += smem[tid + 32]; __syncwarp();
      smem[tid] = x; __syncwarp();
    }
    x += smem[tid + 16]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 8]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 4]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 2]; __syncwarp();
    smem[tid] = x; __syncwarp();
    x += smem[tid + 1]; __syncwarp();
    smem[tid] = x; __syncwarp();
}
// Note: using blockSize as a template arg can benefit from NVCC compiler optimization, 
// which is better than using blockDim.x that is known in runtime.
template<int blockSize>
__global__ void reduce_v4(float *d_in,float *d_out){
    __shared__ float smem[blockSize];

    unsigned int tid = threadIdx.x;
    // 仅改动这里  每个线程加载两个元素到shared mem对应位置
    unsigned int gtid = blockIdx.x * (blockSize*2) + threadIdx.x;
    // load: 每个线程加载两个元素到shared mem对应位置
    smem[tid] = d_in[gtid] + d_in[gtid + blockSize];
    __syncthreads();

    // compute: reduce in shared mem
    // 思考这里是如何并行的
    for (unsigned int index = blockDim.x / 2; index > 32; index >>= 1) {
        if (tid < index) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    // __syncthreads 耗时多，最后一个warp 不需要
    // last warp 32拎出来单独作reduce
    if (tid < 32) {
        WarpSharedMemReduce(smem, tid);
    }
    // store: write back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(float *out, float groudtruth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    //const int N = 32 * 1024 * 1024;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
    //int GridSize = 100000;
    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float*)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize / 2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v4<blockSize / 2><<<Grid,Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        //for(int i = 0; i < GridSize;i++){
            //printf("res per block : %lf ",out[i]);
        //}
        //printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_v3 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
