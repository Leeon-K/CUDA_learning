
// naive m * n matrix transpose
// grid(m / 256, n / 256)  block(256, 256)
__global__ void transpose_naive(float *odata, float *idata, int m, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < m)
    {
        int index_in = row * n + col;
        int index_out = col * m + row; // uncoalseced global memory access
        odata[index_out] = idata[index_in];
    }
}

// share_mem 1
__global__ void transposeOpt1(float *input, float *output, int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float sdata[32][32]; // 32个bank，每个bank 4bytes 1float = 4bytes
    if (col < n && row < m) {
        int index_in = row * n + col;
        sdata[threadIdx.y][threadIdx.x] = input[index_in];
        __syncthreads();
        int dst_col = blockIdx.x * blockDim.x + threadIdx.x;
        int dst_row = blockIdx.y * blockDim.y + threadIdx.y;
        output[dst_row * m + dst_col] = sdata[threadIdx.x][threadIdx.y]; // 
    }
}

// sharbe_mem solve bank conflict
// block( 32 64 ) 
__global__ void transposeOpt1(float *input, float *output, int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // padding = 1
    __shared__ float sdata[32][33]; // 32个bank，每个bank 4bytes 1float = 4bytes
    if (col < n && row < m) {
        int index_in = row * n + col;
        sdata[threadIdx.y][threadIdx.x] = input[index_in];
        __syncthreads();
        int dst_col = blockIdx.x * blockDim.x + threadIdx.x;
        int dst_row = blockIdx.y * blockDim.y + threadIdx.y;
        output[dst_row * m + dst_col] = sdata[threadIdx.x][threadIdx.y]; // 
    }
}
