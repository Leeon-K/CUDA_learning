## GEMV

### GEMV实现 v0 native：
- 原始实现：一个block负责一行，分配256/1024 threads/p；然后reduce;
记录一个神奇的bug，
```
CUDA Error :
File:Line:
gemv 1.cu
Error code: 716
Error text: misaligned address
```
Your problem is that the alignment for float4 is higher than that for float2. Therefore the lines;
Memcpy时 两者数据类型不同 映射到fp上导致的。

#### GEMV实现 v1 优化1，向量化读写，fp4

