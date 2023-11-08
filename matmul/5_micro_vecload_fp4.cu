#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define M 4096
#define N 4096
#define K 1024
// 分块大小
#define BM 32
#define BN 32
#define BK 32
// #define A(i,j) A[(i) + (j)*lda]
// #define B(i,j) B[(i) + (j)*ldb]
// #define C(i,j) C[(i) + (j)*ldc]
#define IDX2C(i, j, ld) ((j) * (ld) + (i)) // columb-major
#define IDX2R(i, j, lr) ((i) * (lr) + (j)) // row-major
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
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A[IDX2C(bx<<5,0,m)]; // blockdim(32,32) 
    B = &B[IDX2C(0,by<<5,k)]; // A定位行， B定位列
    C = &C[IDX2C(bx<<5,by<<5,m)];
    float sum = 0.0;
    for (int i = 0; i < k; i++){
        sum += A[IDX2C(tx,i,m)] * B[IDX2C(i,ty,k)];// 
    }
    C[IDX2C(tx,ty,m)] = alpha * sum + beta * C[IDX2C(tx,ty,m)];
}

__global__  __launch_bounds__(256)
__global__ void gemm_shared_mircokernel_fp4(const int m,const int n,const int k,const float alpha, const float *A, const float *B, const float beta, float* C)
{
    // 分配共享内存  
    __shared__ float sa[BM*BK];
    __shared__ float sb[BK*BN];

    int tx = threadIdx.x; // 一个block有256个线程
    int bx = blockIdx.x, by = blockIdx.y;
    // int row = tx&31, col = tx>>5; // 低位行x  高维列y
    int row1 = (tx&7)<<2, row2 = row1 + 1,row3 = row1 + 2,row4 = row1 + 3;
    int col = tx>>3; // 3位是row 每一个thread  4x1 micro kernel； 5位是col
    A = &A[IDX2C(bx<<5,0,m)]; // blockdim(32,32)
    B = &B[IDX2C(0,by<<5,k)];
    C = &C[IDX2C(bx<<5,by<<5,m)];
    float4 Av, Bv, Cv, Cres;
    Cres.x = 0., Cres.y = 0., Cres.z = 0., Cres.w = 0.;
    float b00;
    for (int i = 0; i < k; i += BK){
        // 存储是 列主序, vec load
        Av = *((float4 *)(&A[IDX2C(row1,col,m)])); // 类型强转
        Bv = *((float4 *)(&B[IDX2C(row1,col,k)]));
        ((float4 *)sa)[tx] = Av;  // 一次读取四个
        sb[IDX2C(col,row1,BK)] = Bv.x; // 一次读取四个
        sb[IDX2C(col,row2,BK)] = Bv.y; 
        sb[IDX2C(col,row3,BK)] = Bv.z; 
        sb[IDX2C(col,row4,BK)] = Bv.w; 
        A += m<<5; // 一行 一次32行
        B += 32; //小方块一列32
        __syncthreads();
        // #pragma unroll
        for (int b_k = 0; b_k < BK; b_k++){
            b00 = sb[IDX2C(col,b_k,BK)];
            // 不能向量化乘加 只能load write
            Cres.x += sa[IDX2C(row1,b_k,BM)] * b00;
            Cres.y += sa[IDX2C(row2,b_k,BM)] * b00; 
            Cres.z += sa[IDX2C(row3,b_k,BM)] * b00; 
            Cres.w += sa[IDX2C(row4,b_k,BM)] * b00; 
        }
        __syncthreads();
    }
    Cv = *((float4 *)(&C[IDX2C(row1,col,m)])); // 向量化读
    Cres.x = alpha * Cres.x + beta * Cv.x;
    Cres.y = alpha * Cres.y + beta * Cv.y;
    Cres.z = alpha * Cres.z + beta * Cv.z;
    Cres.w = alpha * Cres.w + beta * Cv.w;
    *((float4 *)(&C[IDX2C(row1,col,m)])) = Cres;// 向量化写
}

void gpuSgemm(int m, int n, int k, const float *alpha, 
    const float *A, const float *B, const float *beta, float *C) {
        int blocksize = 256;
        // int GridSize = ceil(sqrt((N+bs-1.) / bs));
        // int GridSize = ceil((M*N+blocksize-1.) / blocksize);
        int gridx = floor(M/32);
        int gridy = floor(N/32);
        dim3 Grid(gridx, gridy); //
        dim3 Block(256); // 32 * 32 = 1024  
        //malloc on device
        float *devPtrA, *devPtrB, *devPtrC,*devPtrD;
        cudaMalloc((void**)&devPtrA, sizeof(float) * m * k);
        cudaMalloc((void**)&devPtrB, sizeof(float) * k * n);
        cudaMalloc((void**)&devPtrC, sizeof(float) * m * n);
        cudaMalloc((void**)&devPtrD, sizeof(float) * m * n);
        //copy A and B to device
        cudaMemcpy(devPtrA, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(devPtrB, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
// ------------------------------------------------------------------------------------
        gemm_shared_mircokernel_fp4<<<Grid,Block>>>(m,n,k,*alpha,devPtrA,devPtrB,*beta,devPtrC);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gpu with gemm_shared_v2 kernel time:%f ms\n",milliseconds);
        float* matrix_out_cpu=(float*)malloc(sizeof(float) * M * N);
        float* matrix_out_gpu=(float*)malloc(sizeof(float) * M * N);
        cudaMemcpy(matrix_out_cpu, devPtrC, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        dim3 Grid_n(gridx, gridy); //
        dim3 Block_n(32,32); // 32 * 32 = 1024  
        naive_matmul<<<Grid_n,Block_n>>>(m,n,k,*alpha,devPtrA,devPtrB,*beta,devPtrD);
        cudaMemcpy(matrix_out_gpu, devPtrD, m * n * sizeof(float), cudaMemcpyDeviceToHost);

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

        //release memory on device
        cudaFree(devPtrA);
        cudaFree(devPtrB);
        cudaFree(devPtrC);
        cudaFree(devPtrD);
        free(matrix_out_cpu);
        free(matrix_out_gpu);
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
    
    // // record cpu execution time
    // start=clock();
    // cpuSgemm(M, N, K, &a, matrix_in1, matrix_in2, &b, matrix_out_cpu);
    // stop=clock();
    // duration=(double)(stop-start)/CLOCKS_PER_SEC;
    // printf("cpu time:%f\n",duration);

    ///////////////////////////////////////////////////////////////////////////////////
    gpuSgemm(M, N, K, &a, matrix_in1, matrix_in2, &b, matrix_out_gpu);
  
    // float EPSILON = 0.1;
    // // check result                                             
    // printf("check\n");
    // for (int i = 0; i < M * N; ++i) {
    //     float error = (matrix_out_cpu[i] - matrix_out_gpu[i]) 
    //         / matrix_out_gpu[i];
    //     if (error < -EPSILON || error > EPSILON)
    //         printf("wrong, %f, %f, %f\n", matrix_out_cpu[i], matrix_out_gpu[i], 
    //             error);
    // }
    // printf("right\n");

    //release memory on host
    free(matrix_in1);
    free(matrix_in2);
    free(matrix_out_cpu);
    free(matrix_out_gpu);

    return 0;
}