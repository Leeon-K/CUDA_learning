#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define M 4096
#define N 4096
#define K 4096

#define IDX2C(i,j, ld) (j * ld + i) // column-major

void cpuSgemm(const int m,const int n,const int k,const float* alpha, const float *A, const float *B,
    const float *beta, float* C){
    for (int idx_m = 0; idx_m < m; idx_m++){
        for (int idx_n = 0;idx_n < n; idx_n++){
            float sum = 0.0;
            for (int idx_k = 0; idx_k < k; idx_k++){
                sum += A[IDX2C(idx_m, idx_k, m)] * B[IDX2C(idx_k,idx_n,k)];
            }
            C[IDX2C(idx_m,idx_n,m)] = sum* *(alpha) + *(beta) *C[IDX2C(idx_m,idx_n,m)];
        }
    }
}

__global__ void naive_matmul(const int m,const int n,const int k,const float alpha, const float *A, const float *B, const float beta, float* C)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y; // 分配线程到矩阵位置
    if (ty < m && tx < n){ // 线程不超过矩阵大小
        float sum = 0;
        for (int i=0; i < k;i++){
            sum += A[IDX2C(tx,i,m)] * B[IDX2C(i,ty,k)];
        }
        C[IDX2C(tx,ty,m)] = alpha * sum + beta * C[IDX2C(tx,ty,m)];
    }
}


void gpuSgemm(int m, int n, int k, const float *alpha, 
    const float *A, const float *B, const float *beta, float *C) {
        int blocksize = 256;
        // int GridSize = ceil(sqrt((N+bs-1.) / bs));
        // int GridSize = ceil((M*N+blocksize-1.) / blocksize);
        int gridx = floor(M/32);
        int gridy = floor(N/32);
        dim3 Grid(gridx, gridy); //
        dim3 Block(32,32); // 32 * 32 = 1024  
        //malloc on device
        float *devPtrA, *devPtrB, *devPtrC;
        cudaMalloc((void**)&devPtrA, sizeof(float) * m * k);
        cudaMalloc((void**)&devPtrB, sizeof(float) * k * n);
        cudaMalloc((void**)&devPtrC, sizeof(float) * m * n);
        //copy A and B to device
        cudaMemcpy(devPtrA, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(devPtrB, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
// ------------------------------------------------------------------------------------
        naive_matmul<<<Grid,Block>>>(m,n,k,*alpha,devPtrA,devPtrB,*beta,devPtrC);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gpu with naive kernel time:%f ms\n",milliseconds);
  // ----------------------------------------------------------------------------------
        cudaMemcpy(C, devPtrC, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        
        //release memory on device
        cudaFree(devPtrA);
        cudaFree(devPtrB);
        cudaFree(devPtrC);
}

int main(){
    float rand_min = -10.0, rand_max = 10.0, rand_num = 0.0;

    float* matrix_in1 = (float*)malloc(sizeof(float) * M * K);
    float* matrix_in2 = (float*)malloc(sizeof(float) * K * N);
    float* matrix_out_cpu = (float*)malloc(sizeof(float) * M * N);
    float* matrix_out_gpu = (float*)malloc(sizeof(float) * M * N);

    for (int i = 0; i< M * K; i++){
        rand_num = (float)rand() / RAND_MAX; // RAND_MAX = 32767
        matrix_in1[i] = rand_min + rand_num * (rand_max - rand_min);
    }
    for (int i = 0; i < K * N; ++i) {
        rand_num = (float)rand()/RAND_MAX;
        matrix_in2[i] = rand_min + rand_num * (rand_max - rand_min);
    }

    clock_t start, stop;
    float a = 1.0, b = 0.0;
    double duration;
    
    // record cpu execution time
    start=clock();
    cpuSgemm(M, N, K, &a, matrix_in1, matrix_in2, &b, matrix_out_cpu);
    stop=clock();
    duration=(double)(stop-start)/CLOCKS_PER_SEC;
    printf("cpu time:%f\n",duration);

    ///////////////////////////////////////////////////////////////////////////////////
    gpuSgemm(M, N, K, &a, matrix_in1, matrix_in2, &b, matrix_out_gpu);
  
    float EPSILON = 0.1;
    // check result                                             
    printf("check\n");
    for (int i = 0; i < M * N; ++i) {
        float error = (matrix_out_cpu[i] - matrix_out_gpu[i]) 
            / matrix_out_gpu[i];
        if (error < -EPSILON || error > EPSILON)
            printf("wrong, %f, %f, %f\n", matrix_out_cpu[i], matrix_out_gpu[i], 
                error);
    }
    printf("right\n");

    //release memory on host
    free(matrix_in1);
    free(matrix_in2);
    free(matrix_out_cpu);
    free(matrix_out_gpu);

    return 0;
}