#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <string>
#include <stdexcept>

static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

// reduce Op = sum
template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

//warp reduce  
template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T warpReduce(T val){
    for (int offset = 16; offset > 0; offset /= 2){
        // warp原语,直接读取warp内其他lane寄存器的数据
        // __shfl_down_sync 加到 down offset
        val = ReductionOp<T>()(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// blockReduceSum
template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T blockReduce(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int warp_nums = (blockDim.x + 31) / 32; // warp_nums防止为0
    static __shared__ float warpres[64]; // 64 * 32 = 2048 不会超过2048个线程

    // block内的warp结果 保存在每个warp内的0号线程，若lane==0 写入warp res
    val = warpReduce<ReductionOp, T>(val);
    if(lane == 0){
        warpres[warp_id] = val;
    }
    __syncthreads();
    // 每个warp的结果进行reduce 得到--> blockReduce
    float warp_val = tid < warp_nums ? warpres[tid] : 0; //超出的部分为0 
    return warpReduce<ReductionOp, T>(warp_val);
}

// // 把block reduce拆分为多个warp reduce来计算
// template<template<typename> class ReductionOp, typename T>
// __device__ __forceinline__ T blockReduce(T val){
//     int tid = threadIdx.x;
//     int warp_id = tid / 32;
//     int lane_id = tid % 32;
//     // 向上进1，以防分配的线程数量小于32导致warp nums为0
//     int warp_nums = (blockDim.x + 31) / 32;
//     static __shared__ float warpres[64];
//     // block内每个warp reduce的结果，该结果保存在每个warp内的0号线程，所以L65用0号线程写入warp res
//     val = warpReduce<ReductionOp, T>(val);
//     if (lane_id == 0){
//         warpres[warp_id] = val;
//     }
//     __syncthreads();
//     // 最后把每个warp的结果再作一个reduce得到最终一个block的结果
//     float warp_val = tid < warp_nums ? warpres[tid] : 0;
//     return warpReduce<ReductionOp, T>(warp_val);
// }


// // {vec 1*N  mat N*M} = res 1*M ,VECS_PER_THREAD =  (N / THREAD_NUMS) / VEC_SIZE , num_cols=N
// template<int VECS_PER_THREAD, int VEC_SIZE, int THREAD_NUMS>
// __global__ void gemv(T* d_mat, T* d_vec, T* d_dst, int num_cols){
//     // v0 基础版本
//     // 一个grid负责一行，N列 每个block计算得出256个元素相乘的结果
//     // 每个元素直接从global memory中读取
//     int tid = threadIdx.x; // 第几个元素
//     int bid = blockIdx.x;  // 行
//     float sum = 0 = 0.0f;
//     for (int i=0; i<VECS_PER_THREAD; i++){
//         float vec = d_vec[tid * VEC_SIZE]; // vec_offset = tid * VECS_PER_THREAD + i
//         float mat = d_mat[bid*num_cols + tid * VEC_SIZE]; // 
//         sum += vec[i] * mat[i];
//     }

//     float reduce_res = blockReduce<SumOp,float>(sum);

//     if (tid == 0){ //计算完成 每个block的第一个线程负责写入
//         d_dst[bid] = reduce_res;
//     }
//     __syncthreads();
// }


// 一个blk计算一个元素
// mat * vec = {M, N} * {N, 1}/{1, N}
template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(float* d_mat, float* d_vec, float* res, int num_cols) {
    int tid = threadIdx.x; // 第几个元素
    int bid = blockIdx.x;  // 行
    float thread_local_sum = 0.0f;
    for (int i=0; i<VECS_PER_THREAD; i++){
        float* vec = &d_vec[tid * VEC_SIZE]; // vec_offset = tid * VECS_PER_THREAD + i
        float* mat = &d_mat[bid*num_cols + tid * VEC_SIZE]; // 
        thread_local_sum += vec[i] * mat[i];
    }
    //reduce to get the final val
    float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
    //store to gmem
    if(tid == 0) {
        res[blockIdx.x] = reduce_res;
    }
    __syncthreads();
}

// 模板可以根据不同size的数据 设置不同的kernel
template<int VECS_PER_THREAD, int VEC_SIZE, int THREAD_NUMS>
struct DispatchLauncher{
    template<typename T>
    static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N){
        dim3 Grid(M); // max grid (2147483647, 65535, 65535)
        dim3 Block(THREAD_NUMS);
        float milliseconds = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        gemv<VECS_PER_THREAD, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N);
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("[ERROR] CUDA runtime error: ") +  (_cudaGetErrorEnum(result)) + " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
        printf("called\n");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // 等待stop事件完成
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gemv latency = %f ms\n", milliseconds);
    }
}; // 数据类型