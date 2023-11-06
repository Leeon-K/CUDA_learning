#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// cpu time:5.561000
// gpu with blas time:0.750000
// check  M*K  K*N
#define M 4096 
#define N 4096 
#define K 1024
// #define M 32
// #define N 8
// #define K 16
#define EPSILON 0.01
#define IDX2C(i, j, ld) ((j) * (ld) + (i)) // column-major

void cpuSgemm(int m, int n, int k, const float *alpha, const float *A, const float *B,
    const float *beta, float *C) {
    for (int idx_m = 0; idx_m < m; ++idx_m) {
        for (int idx_n = 0; idx_n < n; ++idx_n) {
            float sum = 0.0;
            for (int idx_k = 0; idx_k < k; ++idx_k) {
                sum += A[IDX2C(idx_m, idx_k, m)] * B[IDX2C(idx_k, idx_n, k)];
            }
            C[IDX2C(idx_m, idx_n, m)] = *(alpha) * sum + *(beta) * C[IDX2C(idx_m, idx_n, m)];
        }
    }
}

void gpuBlasSgemm(int m, int n, int k, const float *alpha, 
    const float *A, const float *B, const float *beta, float *C) {
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    //malloc on device
    float *devPtrA, *devPtrB, *devPtrC;
    cudaMalloc((void**)&devPtrA, sizeof(float) * m * k);
    cudaMalloc((void**)&devPtrB, sizeof(float) * k * n);
    cudaMalloc((void**)&devPtrC, sizeof(float) * m * n);
    //copy A and B to device
    cublasSetVector (m * k, sizeof(float), A, 1, devPtrA, 1);
    cublasSetVector (k * n, sizeof(float), B, 1, devPtrB, 1);
    //use clublas to compute
    cudaDeviceSynchronize();
// ----------------------------------------------------------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, devPtrA, m, devPtrB, k, beta, devPtrC, m);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
// ----------------------------------------------------------------------------------
    printf("gpu with blas kernel time:%f ms\n",milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();
    //copy devPtrC to host
    cublasGetVector(m * n, sizeof(float), devPtrC, 1, C, 1);
    //release memory on device
    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    cublasDestroy(handle);
}

int main() {
    srand((unsigned)time(NULL));
    float rand_min = -10.0, rand_max = 10.0, rand_num = 0.0;

    float* matrix_in1 = (float *)malloc(sizeof(float) * M * K);
    float* matrix_in2 = (float *)malloc(sizeof(float) * K * N);
    float* matrix_out_cpu = (float *)malloc(sizeof(float) * M * N);
    float* matrix_out_gpu_blas = (float *)malloc(sizeof(float) * M * N);
    
    for (int i = 0; i < M * K; ++i) {
        rand_num = (float)rand()/RAND_MAX;
        matrix_in1[i] = rand_min + rand_num * (rand_max - rand_min);
    }
    for (int i = 0; i < K * N; ++i) {
        rand_num = (float)rand()/RAND_MAX;
        matrix_in2[i] = rand_min + rand_num * (rand_max - rand_min);
    }

    float a = 2.0, b = 0.0;
    clock_t start, stop;
    double duration;
    
    // record cpu execution time
    // start=clock();
    // cpuSgemm(M, N, K, &a, matrix_in1, matrix_in2, &b, matrix_out_cpu);
    // stop=clock();
    // duration=(double)(stop-start)/CLOCKS_PER_SEC;
    // printf("cpu time:%f\n",duration);
    
    // record gpu with cublas execution time
    start=clock();
    gpuBlasSgemm(M, N, K, &a, matrix_in1, matrix_in2, &b, matrix_out_gpu_blas);
    stop=clock();
    duration=(double)(stop-start)/CLOCKS_PER_SEC;
    printf("gpu with blas time:%f\n",duration);

    // check result                                             
    // printf("check\n");
    // for (int i = 0; i < M * N; ++i) {
    //     float error = (matrix_out_cpu[i] - matrix_out_gpu_blas[i]) 
    //         / matrix_out_gpu_blas[i];
    //     if (error < -EPSILON || error > EPSILON)
    //         printf("wrong, %f, %f, %f\n", matrix_out_cpu[i], matrix_out_gpu_blas[i], 
    //                error);
    // }
    // printf("right\n");
                                                                                             
    //release memory on host
    free(matrix_in1);
    free(matrix_in2);
    free(matrix_out_cpu);
    free(matrix_out_gpu_blas);

    return EXIT_SUCCESS;
}