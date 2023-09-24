#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
// cuda执行流程， 线程层次结构
__global__ void sum(float *x)
{
    int block_id = blockIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    printf("block_id: %d, global_tid: %d, local_tid: %d\n", block_id, global_tid, local_tid);
    x[global_tid] *= 2;
}

int main(){
    int N = 64;
    int nbytes = N * sizeof(float);
    float *dx, *hx;
    /* allocate GPU mem */ //为什么用二级指针
    cudaMalloc((void **)&dx, nbytes);
    /* allocate CPU mem */
    hx = (float*) malloc(nbytes);
    /* init host data */
    printf("hx original: \n");
    for (int i = 0; i < N; i++) {
        hx[i] = i;
        printf("%g\n", hx[i]);
    }
    /* copy data hx->dx to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    // GPU kernel/ 1, N 1个block，每个block有N个线程
    /* copy data dx /->hx to CPU */
    sum<<<2, N/2>>>(dx);
    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
    printf("hx current: \n");
    for (int i = 0; i<N; i++){
        printf("%g\n", hx[i]);
    }
    cudaFree(dx);
    free(hx);
    
}