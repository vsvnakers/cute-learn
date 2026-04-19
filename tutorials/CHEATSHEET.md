# CUTE 快速参考手册

## 1. Layout 核心 API

```cpp
#include <cute/layout.hpp>

// 创建 Shape
auto shape = make_shape(8, 8);           // 2D: 8x8
auto shape3d = make_shape(4, 8, 16);     // 3D: 4x8x16

// 创建 Stride
auto stride = make_stride(8, 1);         // 行优先
auto stride_c = make_stride(1, 4);       // 列优先

// 创建 Layout
auto layout = make_layout(shape);               // 默认行优先
auto layout2 = make_layout(shape, stride);      // 指定 stride
auto layout3 = make_ordered_layout(shape, 0, 1); // 指定维度顺序

// Layout 操作
layout(i, j);              // 计算 (i,j) 的偏移
layout.shape();            // 获取 Shape
layout.stride();           // 获取 Stride
layout.size();             // 获取总大小
layout.rank();             // 获取维度数

// Slice 操作
auto row = slice<0>(layout, make_coord(5));   // 第 5 行
auto col = slice<1>(layout, make_coord(3));   // 第 3 列
```

## 2. Tensor 核心 API

```cpp
#include <cute/tensor.hpp>

// 创建 Tensor
auto tensor = make_tensor(ptr, layout);

// 访问元素
auto val = tensor(i, j);       // 读
tensor(i, j) = value;          // 写

// Tensor 分片
auto tile = tensor(make_coord(_, _), make_coord(_, _));
```

## 3. Swizzle 核心 API

```cpp
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>

// 创建 Swizzle 函数
auto swz = swizzle<2, 4, 2>{};    // <bank_bits, bank_count_bits, xor_shift>
auto result = swz(offset);         // 应用 Swizzle

// 创建 Swizzle Layout
auto base_layout = make_layout(make_shape(32, 32));
auto swizzled_layout = make_swizzle_layout(
    base_layout,
    swizzle<2, 4, 2>{}
);

// 常见配置
// SM80 共享内存：swizzle<2, 4, 2> 或 swizzle<2, 5, 2>
// SM90 TMA: swizzle<3, 4, 2>
```

## 4. MMA 核心 API

```cpp
#include <cute/atom/mma_atom.hpp>

// SM80 MMA 配置
using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

// 获取 MMA 形状
auto mma_shape = MMA::Shape_MNK{};
auto M = size<0>(mma_shape);   // 16
auto N = size<1>(mma_shape);   // 8
auto K = size<2>(mma_shape);   // 8

// 可用的 MMA 类型
// SM75: SM75_8x8x4_F32F16F16F32_TN
// SM80: SM80_16x8x8_F32F16F16F32_TN
// SM80: SM80_16x8x16_F32F16F16F32_TN
// SM90: SM90_16x8x32_F32F16F16F32_TN
```

## 5. 常用 Layout 模式

### 共享内存矩阵 Layout（避免 Bank Conflict）

```cpp
// 16x16 矩阵，带 Swizzle
auto smem_layout = make_swizzle_layout(
    make_layout(make_shape(16, 16)),
    swizzle<2, 4, 2>{}
);

// 32x32 矩阵，带 Padding
auto padded_layout = make_layout(
    make_shape(32, 33)  // 33 = 32 + 1 padding
);
```

### Vectorized 访问 Layout

```cpp
// 4 元素 vector 访问
auto vec_layout = make_layout(
    make_shape(make_shape(4, 8), 4),  // 4x8 块，每块 4 元素
    make_stride(make_stride(1, 32), 4)
);
```

### Tensor Core MMA Layout

```cpp
// A 矩阵 (16x8)
auto A_layout = make_layout(make_shape(16, 8));

// B 矩阵 (8x8) - 转置存储
auto B_layout = make_layout(make_shape(8, 8), make_stride(1, 8));

// C/D 矩阵 (16x8)
auto C_layout = make_layout(make_shape(16, 8));
```

## 6. 常用数据类型

```cpp
// FP16
#include <cuda_fp16.h>
half x = __float2half(1.0f);
float y = __half2float(x);

// BF16 (SM80+)
#include <cuda_bf16.h>
nv_bfloat16 x = __float2bfloat16(1.0f);

// FP8 (SM90+)
#include <cuda_fp8.h>
__nv_fp8_e4m3 x = __nv_cvt_float_to_fp8(1.0f, __NV_SATFINITE, __NV_E4M3);
```

## 7. 典型 Kernel 模式

### 分块加载模式

```cpp
template<int BLOCK_M, int BLOCK_N>
__global__ void kernel(float* A, float* B, float* C, int M, int N) {
    __shared__ float As[BLOCK_M * BLOCK_N];
    __shared__ float Bs[BLOCK_M * BLOCK_N];

    auto layout = make_layout(make_shape(BLOCK_M, BLOCK_N));
    auto A_tile = make_tensor(As, layout);
    auto B_tile = make_tensor(Bs, layout);

    int tid = threadIdx.x;
    int row = blockIdx.y * BLOCK_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_N + threadIdx.x;

    // 分块处理
    for (int t = 0; t < num_tiles; t++) {
        // 加载
        if (row < M && col < N) {
            A_tile(threadIdx.y, threadIdx.x) = A[row * N + col];
        }
        __syncthreads();

        // 计算
        // ...

        __syncthreads();
    }
}
```

### 流水线模式

```cpp
__global__ void pipeline_kernel(float* A, float* B, float* C) {
    __shared__ float smem[256];

    int tid = threadIdx.x;
    int stage = 0;

    // 预取
    cp_async(smem, A + tid, size<128>());
    cp_async_commit();

    for (int i = 0; i < num_iterations; i++) {
        cp_async_wait_all();
        __syncthreads();

        // 计算当前 stage
        compute(smem);

        // 预取下一阶段
        stage = (stage + 1) % num_stages;
        cp_async(smem + stage * 256, A + tid + i * 256, size<128>());
        cp_async_commit();
    }
}
```

## 8. 调试宏

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CUDA_SYNC() CUDA_CHECK(cudaDeviceSynchronize())
```

## 9. 性能优化清单

- [ ] 使用合适的 block size（通常 128/256/512）
- [ ] 确保全局内存访问合并（coalesced）
- [ ] 使用共享内存减少全局内存访问
- [ ] 避免 Bank Conflict（使用 Swizzle 或 Padding）
- [ ] 使用 Vectorized 访问（float4, etc.）
- [ ] 隐藏内存延迟（流水线、多 issue）
- [ ] 最大化 occupancy（合理的寄存器使用）
- [ ] 使用 Tensor Core（MMA 指令）

## 10. 常见错误排查

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| 编译错误：undefined make_layout | 缺少 include | 添加 `#include <cute/layout.hpp>` |
| 运行时错误：invalid configuration | block size 太大 | 减少 block 维度 |
| 性能差 | Bank Conflict | 使用 Swizzle Layout |
| 结果错误 | 内存未对齐 | 确保 16 字节对齐 |
| 结果错误 | 缺少同步 | 添加 `__syncthreads()` |
