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
// mat * vec = {M, N} * {N, 1}/{1, N}   num_cols = N
template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(float* d_mat, float* d_vec, float* res, int num_cols) {
    int tid = threadIdx.x; // 第几个元素
    int bid = blockIdx.x;  // 列 列主序
    float thread_local_sum = 0.0f;
    // VECS_PER_THREAD = N / THREAD_NUMS / VEC_SIZE = 8
    for (int i=0; i<VECS_PER_THREAD; i++){
        thread_local_sum += d_vec[tid * VEC_SIZE + i * blockDim.x] * d_mat[bid * num_cols + tid * VEC_SIZE + i * blockDim.x];
    }
    //reduce to get the final val
    float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
    //store to gmem
    if(tid == 0) {
        res[blockIdx.x] = reduce_res;
    }
    __syncthreads();
}


// fp4 向量化读写
template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv_fp4(float* d_mat, float* d_vec, float* res, int num_cols) {
    int tid = threadIdx.x; // 第几个元素
    int bid = blockIdx.x;  // 列 列主序
    float thread_local_sum = 0.0f;
    // VECS_PER_THREAD = N / THREAD_NUMS / VEC_SIZE = 8
    for (int i=0; i<VECS_PER_THREAD; i++){
        float4* vec4 = reinterpret_cast<float4*>(&d_vec[tid*VEC_SIZE]);
        float4* mat4 = reinterpret_cast<float4*>(&d_mat[bid*num_cols + tid*VEC_SIZE]);
        int idx = i*blockDim.x;
        thread_local_sum += vec4[idx].x * mat4[idx].x + vec4[idx].y * mat4[idx].y + vec4[idx].z * mat4[idx].z + vec4[idx].w * mat4[idx].w;
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
        // gemv<VECS_PER_THREAD, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N);
        gemv_fp4<VECS_PER_THREAD, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N);
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




// vec * mat, mat is row major
// [1, N] * [N, M]
// logits * v
// 有关fp32/fp16 fma和add的各种重载操作
namespace gemv2 {
    struct half8 {
        half2 h1;
        half2 h2;
        half2 h3;
        half2 h4;

        __device__ half8& operator = (half8 h8) {
            h1 = h8.h1;
            h2 = h8.h2;
            h3 = h8.h3;
            h4 = h8.h4;
            return *this;
        }
    };

    template<int M, typename T>
    struct get_threads_per_mat_row {
        static const int value = M * sizeof(T) / 16;
    };

    inline __device__ float add(float a, float b)
    {
        return a + b;
    }

    inline __device__ float4 add(float4 a, float4 b)
    {
        float4 c;
        c.x = gemv2::add(a.x, b.x);
        c.y = gemv2::add(a.y, b.y);
        c.z = gemv2::add(a.z, b.z);
        c.w = gemv2::add(a.w, b.w);
        return c;
    }
    inline __device__ half add(half a, half b)
    {
        //return __hadd(a, b);
        //if use L216, half+half is not really adding, its so weird, which  cause our result is 32, not 256
        return (half)((float)a+(float)b);
    }

    inline __device__ half2 add(half2 a, half2 b)
    {
        half2 res;
        res.x = gemv2::add(a.x, b.x);
        res.y = gemv2::add(a.y, b.y);
        return res;
    }

    inline __device__ half8 add(half8 a, half8 b)
    {
        half8 c;
        c.h1 = gemv2::add(a.h1, b.h1);
        c.h2 = gemv2::add(a.h2, b.h2);
        c.h3 = gemv2::add(a.h3, b.h3);
        c.h4 = gemv2::add(a.h4, b.h4);
        return c;
    }

    inline __device__ half fma(half a, half b, half c)
    {
        // 有时候编译器会不认识__hmul或者__hadd，所以粗暴转成fp32计算再转回fp16
        return __float2half((float)a * (float)b + (float)c);
    }


    inline __device__ half2 fma(half a, half2 b, half2 c)
    {
        half2 res;
        res.x = gemv2::fma(a, b.x, c.x);
        res.y = gemv2::fma(a, b.y, c.y);
        return res;
    }

    inline __device__ half8 fma(half a, half8 b, half8 c)
    {
        half8 d;
        d.h1 = gemv2::fma(a, b.h1, c.h1);
        d.h2 = gemv2::fma(a, b.h2, c.h2);
        d.h3 = gemv2::fma(a, b.h3, c.h3);
        d.h4 = gemv2::fma(a, b.h4, c.h4);
        return d;
    }

    inline __device__ float fma(float a, float b, float c)
    {
        return a * b + c;
    }

    inline __device__ float4 fma(float a, float4 b, float4 c)
    {
        float4 d;
        d.x = gemv2::fma(a, b.x, c.x);
        d.y = gemv2::fma(a, b.y, c.y);
        d.z = gemv2::fma(a, b.z, c.z);
        d.w = gemv2::fma(a, b.w, c.w);
        return d;
    }
} // namespace gemv2

// fp4 向量化读写
// 1个block处理一个[1, M], 循环处理完[N, M]
// for fp32: <64, M * sizeof(T) / 16 = M / 4, 4>
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
__global__ void gemv_fp4_v2(float* matrix, float* vector, float* res, int N, int M) {
    int tid = threadIdx.x; // 第几个元素
    int mat_o = tid / THREADS_PER_VALUE; // mat offset
    int mat_i = tid % THREADS_PER_VALUE * VEC_SIZE; // mat offset
    int bid = blockIdx.x;  // (行主序)
    // 一个block处理的行数
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    __shared__ float out_smem[512];
    float4 out;
    // 点乘或fma，inter-block循环累加
    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        float4 mat = *reinterpret_cast<float4*>(&matrix[ti * M + mat_i]);
        float logits = vector[ti];
        // fused mul and add: d = a * b + c
        out = gemv2::fma(logits, mat, out);
    }
    // intra-block二分法相加得最终结果
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<float4*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;
        }
        __syncthreads();
        if (mat_o < midpoint) {
            // ROW_PER_ITER中上半部分out和下半部分out相加
            out = gemv2::add(*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i]), out);
        }
        __syncthreads();
    }
    // 二分法最终结果存在首行，写回显存
    if (mat_o == 0) {
        *reinterpret_cast<float4*>(&res[mat_i]) = out;
    }
    __syncthreads();
}

// 模板可以根据不同size的数据 设置不同的kernel

template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
struct DispatchLauncher2
{
    template<typename T>
    static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N){
        dim3 Grid(1);
        dim3 Block(THREADS_PER_BLOCK);
        float milliseconds = 0;
        // 使用cudaevent计时，开销最小
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        // 启动cuda kernel
        gemv_fp4_v2<THREADS_PER_BLOCK, THREADS_PER_VALUE, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N, M);
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("[ERROR] CUDA runtime error: ") +  (_cudaGetErrorEnum(result)) + " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
        printf("called\n");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gemv latency = %f ms\n", milliseconds);
    }
};