#include "bits/stdc++.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include <stdio.h>
using namespace std;

//latency =  0.542720 ms
template<int blockSize>
__global__ void reduce_v0(float *d_in,float *d_out){
    // for (int i=0;i<blockSize;i++){
    //     if (i<2){printf("d_in[%d]: %f\n", i, d_in[i]);
    // }
    __shared__ float smem[blockSize]; // blocksize = 256

    int tid = threadIdx.x;
    // int gtid = blockIdx.x * blockSize + threadIdx.x;
    // load: 每个线程加载一个元素到shared mem对应位置
    smem[tid] = d_in[tid];
    __syncthreads(); // 把sharememory同步一下

    // compute: reduce in shared mem
    // 引入了 shared memory
    for(int index = 1; index < blockDim.x; index *= 2) {
        if (tid % (2 * index) == 0) { // 第一轮只有0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30这些线程会执行 // 后面0 4 8 16 32 64 128 256
            smem[tid] += smem[tid + index];
            // printf("tid: %d,index %d, smem[tid]: %f, smem[tid + index]: %f\n", tid,index, smem[tid], smem[tid + index]);
        }
        __syncthreads();
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
    const int N = 25600000;
    // const int N = 257;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    // const int blockSize = 4;
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
    // a[N-1] = 2.0f
    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v0<blockSize><<<Grid,Block>>>(d_a, d_out);
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
    printf("reduce_v0 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
