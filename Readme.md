## Reduce优化
#### baseline
```
for (size_t i = 0; i < n; ++i) {
      sum += input[i];
    }
```
#### v0  并行化处理 引入了 shared memory
```
template<int blockSize>
__global__ void reduce_v0(float *d_in,float *d_out){
    __shared__ float smem[blockSize]; // blocksize = 256

    int tid = threadIdx.x;
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
```
#### v1 消除warp divergence（index 123的） 
每个warp内线程一起干活

#### v2 避免bank conflict 
常用：padding
#### v3 
