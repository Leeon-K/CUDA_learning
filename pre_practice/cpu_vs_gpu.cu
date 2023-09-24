#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Windows.h>
#include <time.h>
#ifdef WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
#ifdef WIN32
int gettimeofday(struct timeval *tp, void *tzp)
{
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;
  GetLocalTime(&wtm);
  tm.tm_year   = wtm.wYear - 1900;
  tm.tm_mon   = wtm.wMonth - 1;
  tm.tm_mday   = wtm.wDay;
  tm.tm_hour   = wtm.wHour;
  tm.tm_min   = wtm.wMinute;
  tm.tm_sec   = wtm.wSecond;
  tm. tm_isdst  = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;
  return (0);
}
#endif
using namespace std;
// CPU version
int cpu_v()
{
    struct timeval start, end;
    gettimeofday( &start, NULL );
    float*A, *B, *C;
    int n = 1024 * 1024;
    int size = n * sizeof(float);
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    for(int i=0;i<n;i++)
    {
        A[i] = 90.0;
        B[i] = 10.0;
    }
    
    for(int i=0;i<n;i++)
    {
        C[i] = A[i] + B[i];
    }

    float max_error = 0.0;
    for(int i=0;i<n;i++)
    {
        max_error += fabs(100.0-C[i]);
    }
    cout << "max_error is " << max_error << endl;
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "total time is " << timeuse/1000 << "ms" <<endl;
    return 0;
}

// GPU version block 中的 thread 个数为 1024 充分利用SM
__global__ void Plus(float A[], float B[], float C[], int n)  // host 端调用 kernel 函数时，需要指定 grid 和 block 的大小
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

int gpu_v()
{
    struct timeval start, end;
    gettimeofday( &start, NULL );
    float*A, *Ad, *B, *Bd, *C, *Cd;
    int n = 1024 * 1024;
    int size = n * sizeof(float);

    // CPU端分配内存
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // 初始化数组
    for(int i=0;i<n;i++)
    {
        A[i] = 90.0;
        B[i] = 10.0;
    }
    // GPU端分配内存
    cudaMalloc((void**)&Ad, size);
    cudaMalloc((void**)&Bd, size);
    cudaMalloc((void**)&Cd, size);

    
    // CPU的数据拷贝到GPU端
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
    // 定义kernel执行配置，（1024*1024/1024）个block，每个block里面有1024个线程
    // 线程多反而慢 因为
    dim3 dimBlock(512);
    dim3 dimGrid(n/512);

    // 执行kernel
    Plus<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, n);

    // 将在GPU端计算好的结果拷贝回CPU端
    cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

    // 校验误差
    float max_error = 0.0;
    for(int i=0;i<n;i++)
    {
        max_error += fabs(100.0 - C[i]);
    }

    cout << "max error is " << max_error << endl;

    // 释放CPU端、GPU端的内存
    free(A);
    free(B);
    free(C);
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "total time is " << timeuse/1000 << "ms" <<endl;
    return 0;
}


int main()
{
    // cpu_v();
    gpu_v();
}