#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define M 2048
#define N 2048
#define K 2048
// 分块大小
#define BM 128
#define BN 128
#define BK 8
// #define A(i,j) A[(i) + (j)*lda]
// #define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define IDX2C(i, j, ld) ((j) * (ld) + (i)) // columb-major
#define IDX2R(i, j, lr) ((i) * (lr) + (j)) // row-major
// #define vload(v1,addr) v1 = *((float4 *)(addr));
// #define vstore(addr,v1) *((float4 *)(addr)) = v1;
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;
//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
//v1 = alpha * v2 + beta * v3, simd fma
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;

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
    B = &B[IDX2C(0,by<<5,k)];
    C = &C[IDX2C(bx<<5,by<<5,m)];
    float sum = 0.0;
    for (int i = 0; i < k; i++){
        sum += A[IDX2C(tx,i,m)] * B[IDX2C(i,ty,k)];
    }
    C[IDX2C(tx,ty,m)] = alpha * sum + beta * C[IDX2C(tx,ty,m)];
}

__global__ void gemm_submat_warppara(const int m,const int n,const int k,const float alpha, const float *A, const float *B, const float beta, float* C)
{
    // 分配共享内存  
    __shared__ float sa[1024];
    __shared__ float sb[1024];
    int ldc = m;
    int tx = threadIdx.x; // 一个block有256个线程
    int bx = blockIdx.x, by = blockIdx.y;
    int warp_id = tx>>5; // 256 分为 8个warp
    int lane_id = tx&31;
    int warp_row = warp_id & 3, warp_col = warp_id >> 2; // warp级别的并行
    int row_w = lane_id&3, col_w = lane_id>>2;
    
    int row_c = (warp_row<<5) + (row_w<<3), col_c = (warp_col<<6) + (col_w<<3);
    int row_a = (tx&31)<<2, col_a = tx>>5;
    int row_b = (tx&1)<<2, col_b = tx>>1;
    
    A = &A[IDX2C(bx<<7,0,m)]; //分为 64 64 的小块
    B = &B[IDX2C(0,by<<7,k)];
    C = &C[IDX2C(bx<<7,by<<7,m)];
    float4 Av1, Av2, Bv1, Bv2,  Cv[16], Cres[16];
    memset(Cres, 0, sizeof(Cres));
    for (int i = 0; i < k; i += BK){
        // 存储是 列主序, vec load
        vload(Av1, &A[IDX2C(row_a, col_a, m)]);
        vload(Bv1, &B[IDX2C(row_b, col_b, k)]);
        ((float4 *)sa)[tx] = Av1;  // sa转换为float4 一次读取四个
        sb[IDX2C(col_b,row_b,BN)] = Bv1.x; // 一次读取四个
        sb[IDX2C(col_b,row_b+1,BN)] = Bv1.y; 
        sb[IDX2C(col_b,row_b+2,BN)] = Bv1.z; 
        sb[IDX2C(col_b,row_b+3,BN)] = Bv1.w; 
        A += m << 3;
        B += 8; // 一次8行 
        __syncthreads();
        // #pragma unroll
        for (int b_k = 0; b_k < BK; b_k++){
            // vload(Av, &sa[IDX2C(row_a, b_k, BM)]) // 行分块
            // vload(Bv, &sb[IDX2C(col_c, b_k, BN)]) // 列分块
            // vscal(Cres[0], Av, Bv.x)
            // vscal(Cres[1], Av, Bv.y)
            // vscal(Cres[2], Av, Bv.z)
            // vscal(Cres[3], Av, Bv.w)
            vload(Av1, &sa[IDX2C(row_c, b_k, BM)]) // 行分块
            vload(Av2, &sa[IDX2C(row_c+4, b_k, BM)]) // 列分块
            vload(Bv1, &sb[IDX2C(col_c, b_k, BN)]) // 行分块
            vload(Bv2, &sb[IDX2C(col_c+4, b_k, BN)]) // 列分块
            vscal(Cres[0], Av1, Bv1.x)
            vscal(Cres[1], Av2, Bv1.x)
            vscal(Cres[2], Av1, Bv1.y)
            vscal(Cres[3], Av2, Bv1.y)
            vscal(Cres[4], Av1, Bv1.z)
            vscal(Cres[5], Av2, Bv1.z)
            vscal(Cres[6], Av1, Bv1.w)
            vscal(Cres[7], Av2, Bv1.w)
            vscal(Cres[8], Av1, Bv2.x)
            vscal(Cres[9], Av2, Bv2.x)
            vscal(Cres[10], Av1, Bv2.y)
            vscal(Cres[11], Av2, Bv2.y)
            vscal(Cres[12], Av1, Bv2.z)
            vscal(Cres[13], Av2, Bv2.z)
            vscal(Cres[14], Av1, Bv2.w)
            vscal(Cres[15], Av2, Bv2.w)
        }
        __syncthreads();
    }
    vload(Cv[0], &C[IDX2C(row_c,col_c,m)])
    vload(Cv[1], &C[IDX2C(row_c+4,col_c,m)])
    vload(Cv[2], &C[IDX2C(row_c,col_c+1,m)])
    vload(Cv[3], &C[IDX2C(row_c+4,col_c+1,m)]) // 向量化读
    vload(Cv[4], &C[IDX2C(row_c,col_c+2,m)])
    vload(Cv[5], &C[IDX2C(row_c+4,col_c+2,m)])
    vload(Cv[6], &C[IDX2C(row_c,col_c+3,m)])
    vload(Cv[7], &C[IDX2C(row_c+4,col_c+3,m)]) // 向量化读
    vload(Cv[8], &C[IDX2C(row_c,col_c+4,m)])
    vload(Cv[9], &C[IDX2C(row_c+4,col_c+4,m)]) 
    vload(Cv[10], &C[IDX2C(row_c,col_c+5,m)])
    vload(Cv[11], &C[IDX2C(row_c+4,col_c+5,m)])
    vload(Cv[12], &C[IDX2C(row_c,col_c+6,m)])
    vload(Cv[13], &C[IDX2C(row_c+4,col_c+6,m)])
    vload(Cv[14], &C[IDX2C(row_c,col_c+7,m)])
    vload(Cv[15], &C[IDX2C(row_c+4,col_c+7,m)])
    
    for (int i = 0; i < 16; i++){
        simd_axpby(Cres[i],alpha,Cres[i],beta,Cv[i])
    }

    vstore(&C(row_c,col_c), Cres[0])
    vstore(&C(row_c+4,col_c), Cres[1])
    vstore(&C(row_c,col_c+1), Cres[2])
    vstore(&C(row_c+4,col_c+1), Cres[3])
    vstore(&C(row_c,col_c+2), Cres[4])
    vstore(&C(row_c+4,col_c+2), Cres[5])
    vstore(&C(row_c,col_c+3), Cres[6])
    vstore(&C(row_c+4,col_c+3), Cres[7])
    vstore(&C(row_c,col_c+4), Cres[8])
    vstore(&C(row_c+4,col_c+4), Cres[9])
    vstore(&C(row_c,col_c+5), Cres[10])
    vstore(&C(row_c+4,col_c+5), Cres[11])
    vstore(&C(row_c,col_c+6), Cres[12])
    vstore(&C(row_c+4,col_c+6), Cres[13])
    vstore(&C(row_c,col_c+7), Cres[14])
    vstore(&C(row_c+4,col_c+7), Cres[15])
}

void gpuSgemm(int m, int n, int k, const float *alpha, 
    const float *A, const float *B, const float *beta, float *C) {
        int blocksize = 256;
        // int GridSize = ceil(sqrt((N+bs-1.) / bs));
        // int GridSize = ceil((M*N+blocksize-1.) / blocksize);
        int gridx = floor(m/BM);
        int gridy = floor(n/BN);
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
        gemm_submat_warppara<<<Grid,Block>>>(m,n,k,*alpha,devPtrA,devPtrB,*beta,devPtrC);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gpu with gemm_shared_v2 kernel time:%f ms\n",milliseconds);
        float* matrix_out_cpu=(float*)malloc(sizeof(float) * M * N);
        float* matrix_out_gpu=(float*)malloc(sizeof(float) * M * N);
        cudaMemcpy(matrix_out_cpu, devPtrC, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        dim3 Grid_n(m/32, n/32); //
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