
// naive m * n matrix transpose
// grid(m, n)  block(256, 256)
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

// 