#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;
// typedef float FLOAT;
// kernel func vec_add<<grid, bs>>>
__global__ void vec_add(float *x, float *y, float *z, int n)
{
    /* 2D grid */ //2d  blockDim.x 一个bloak有几个thread；  gridDim：一个grid有几个block
    // int idx = ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x)
    int idx = (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x);
    /* 1D grid */
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // printf("idx:%d, blockDim.x:%d, blockIdx.x:%d, threadIdx.x:%d\n", idx, blockDim.x, blockIdx.x, threadIdx.x);
    if (idx < n)
        z[idx] = x[idx] + y[idx];
}

void vec_add_cpu(float *x, float *y,float *z,int N)
{
    for (int i = 0; i < N; i++)
    {
        z[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 10000;
    int nbytes = N * sizeof(float);

    int bs = 256;

    // 2d grid
    int s = ceil(sqrt((N+bs-1.) / bs));  // N+bs-1.向上取整
    dim3 grid(s, s);

    // // 1d grid
    // int s = ceil((N+bs-1.) / bs);
    // dim3 grid(s);

    float *dx, *hx;
    float *dy, *hy;
    float *dz, *hz;
    
    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);

    float milliseconds = 0;

    // allocate CPU mem
    hx = (float *)malloc(nbytes);
    hy = (float *)malloc(nbytes); 
    hz = (float *)malloc(nbytes);

    for (int i = 0; i < N; i++)
    {
        hx[i] = 1.0f;
        hy[i] = 2.0f;
    }

    // copy data to GPU
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // launch GPU kernel
    vec_add<<<grid, bs>>>(dx, dy, dz, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // 同步 
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);

    float* hz_cpu_res = (float *) malloc(nbytes);
    vec_add_cpu(hx, hy, hz_cpu_res, N);

    /* check GPU result with CPU*/
    for (int i = 0; i < N; ++i) {
        if (fabs(hz_cpu_res[i] - hz[i]) > 1e-6) {
            printf("Result verification failed at element index %d!\n", i);
        }
    }
    printf("Result right\n");
    printf("Mem BW= %f (GB/sec)\n", (float)N*4/milliseconds/1e6);///1.78gb/s
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu_res);

    return 0;
}