#include "gemv.cuh"

// 行主序 mat (row major) {N*M} 
// vec {1*N}  mat {N*M}  res {1*M}
template <typename T>
void gemvCPU(T* mat, T* vec, T* res, int M, int N) {
  for (int i = 0; i < M; i++) {
    T sum = 0;
    for (int j = 0; j < N; j++) {
      sum += mat[i + j * M] * vec[j]; 
    }
    res[i] = sum;
    if (i < 5)
        {
            printf("cpu res = %f\n", res[i]);
        }
  }
}


template <typename T>
bool CheckResult(const T *out,const T *groudtruth,const int N) {
  printf("Checking: %d size vec\n", N);
  for (int i = 0; i < N; i++) {
    if (out[i] != groudtruth[i]) {
      printf("%dth wrong; res is %f, groudtruth is %f\n", i, out[i], groudtruth[i]);
      return false;
    }
  }
  printf("CheckResult: all right %f", out[0]);
  return true;
}

template <typename T>
void gemv_kernel(T* vec,
                 T* d_vec,
                 T* mat,
                 T* d_mat,
                 T* dst,
                 T* d_dst
                 ) {
    constexpr int N = 2048;//256 * 8
    constexpr int M = 256;

//  initialize<T>(vec, d_vec, mat, d_mat, dst, d_dst, M, N);
    vec = (T *)malloc(N * sizeof(T));
    cudaMalloc((void **)&d_vec, N * sizeof(T));

    mat = (T *)malloc(M * N * sizeof(T));
    cudaMalloc((void **)&d_mat, M * N * sizeof(T));

    dst = (T*)malloc(M * sizeof(T));
    cudaMalloc((void **)&d_dst, M * sizeof(T));

    for(int i = 0; i < N; i++){
        vec[i] = (T)1;
    }
    for(int i = 0; i < N * M; i++){
        mat[i] = (T)1;
    }
    // 都是1，可能有错误的算法也给算对了；改成2，就能看出来了
    vec[1000] = 2;
    mat[10000] = 2;

    // gemvCPU(mat, vec, dst, M, N);

    cudaMemcpy(d_vec, vec, N * sizeof(T), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mat, mat, M * N * sizeof(T), cudaMemcpyHostToDevice);
    constexpr int THREADS_PER_BLOCK = 256; // 每个block 256个线程（max=1024）
    // constexpr int VEC_SIZE = 1; // fp32向量长度 vec_size = 4 -->fp4 ;= 1
    constexpr int VEC_SIZE = 4;
    constexpr int THREADS_PER_VALUE = M * sizeof(T) / 16; // ?
    // dispatch kernel 
    DispatchLauncher2<THREADS_PER_BLOCK,THREADS_PER_VALUE, VEC_SIZE>::template launcher<T>(d_mat, d_vec, d_dst, M, N);
    
    CHECK(cudaMemcpy(dst, d_dst, M * sizeof(T), cudaMemcpyDeviceToHost));
    // cudaMemcpy(dst, d_dst, M * sizeof(T), cudaMemcpyDeviceToHost);
    T* groudtruth = (T*)malloc(sizeof(T) * M);
    // gemvCPU(mat, vec, groudtruth, M, N);
    // 注意：cpu函数里面将输入类型强转为fp32，因为cpu没有fp16类型
    float* fp32_mat = reinterpret_cast<float*>(mat);
    float* fp32_vec = reinterpret_cast<float*>(vec);
    float* fp32_groudtruth = reinterpret_cast<float*>(groudtruth);
    gemvCPU<float>(fp32_mat, fp32_vec, fp32_groudtruth, M, N);
    
    float* fp32_dst = reinterpret_cast<float*>(dst);
    bool is_right = CheckResult(fp32_dst, fp32_groudtruth, M);
    
    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_dst);
    free(vec);
    free(mat);
    free(dst);
}

template void gemv_kernel<float>(float*, float*, float*, float*, float*, float*);

int main() {
    float *vec;
    float *d_vec;
    float *mat;
    float *d_mat;
    float *res;
    float *d_res;
    gemv_kernel<float>(vec, d_vec, mat, d_mat, res, d_res);
}