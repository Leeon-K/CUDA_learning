
__global__ void gemm_submat4x4(const int m,const int n,const int k,const float alpha, const float *A, const float *B, const float beta, float* C)
{
    // 分配共享内存  
    __shared__ float sa[1024];
    __shared__ float sb[1024];

    int tx = threadIdx.x; // 一个block有32*32个线程
    int bx = blockIdx.x, by = blockIdx.y;
    // int row = tx&31, col = tx>>5; // 低位行x  高维列y
    int row_a = (tx & 15) << 2, col_a = tx >> 4; // 
    int row_b = (tx & 3) << 2, col_b = tx >> 2; // row_b 是thread连续的
    int col_c = col_a<<2;
    A = &A[IDX2C(bx<<6,0,m)]; // blockdim(32,32)
    B = &B[IDX2C(0,by<<6,k)];
    C = &C[IDX2C(bx<<6,by<<6,m)];
    float4 Av, Bv, Cv[4], Cres[4];
    memset(Cres, 0, sizeof(Cres));
    for (int i = 0; i < k; i += BK){
        // 存储是 列主序, vec load
        vload(Av, &A[IDX2C(row_a, col_a, m)]);
        vload(Bv, &B[IDX2C(row_b, col_b, k)]);
        ((float4 *)sa)[tx] = Av;  // 一次读取四个
        sb[IDX2C(col_b,row_b,BK)] = Bv.x; // 一次读取四个
        sb[IDX2C(col_b,row_b+1,BK)] = Bv.y; 
        sb[IDX2C(col_b,row_b+2,BK)] = Bv.z; 
        sb[IDX2C(col_b,row_b+3,BK)] = Bv.w; 
        A += m << 4;
        B += 16; // 一次16行 
        __syncthreads();
        #pragma unroll
        for (int b_k = 0; b_k < BK; b_k++){
            vload(Av, &sa[IDX2C(row_a, b_k, BM)])
            vload(Bv, &sb[IDX2C(col_c, b_k, BK)])
            vscal(Cres[0], Av, Bv.x)
            vscal(Cres[1], Av, Bv.y)
            vscal(Cres[2], Av, Bv.z)
            vscal(Cres[3], Av, Bv.w)
        }
        __syncthreads();
    }
    vload(Cv[0], &C[IDX2C(row_a,col_c,m)])
    vload(Cv[1], &C[IDX2C(row_a,col_c+1,m)])
    vload(Cv[2], &C[IDX2C(row_a,col_c+2,m)])
    vload(Cv[3], &C[IDX2C(row_a,col_c+3,m)]) // 向量化读
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    vstore(&C[IDX2C(row_a,col_c, m)], Cres[0])
    vstore(&C[IDX2C(row_a,col_c + 1, m)], Cres[1])
    vstore(&C[IDX2C(row_a,col_c + 2, m)], Cres[2])
    vstore(&C[IDX2C(row_a,col_c + 3, m)], Cres[3])  // 向量化写
}