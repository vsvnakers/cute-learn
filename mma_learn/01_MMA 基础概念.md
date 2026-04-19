# 第一课：MMA 基础概念

## 1.1 什么是 MMA？

**MMA (Matrix Multiply-Accumulate)** 是矩阵乘累加操作的硬件加速单元。

### MMA 的数学定义

```
D = A × B + C

其中:
- A: M×K 矩阵
- B: K×N 矩阵
- C: M×N 矩阵（累加器）
- D: M×N 矩阵（输出）
```

### 为什么需要 MMA？

**传统 CUDA Core 方式**：
```cuda
// 16×8×8 矩阵乘法需要 16×8×8 = 1024 次 FMA
for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) {
        for (int k = 0; k < 8; k++) {
            C[i][j] += A[i][k] * B[k][j];  // 1024 次迭代
        }
    }
}
```

**MMA 方式**：
```cuda
// 1 条 MMA 指令完成 16×8×8 矩阵乘法
mma.sync.aligned.m16n8k8... {d}, {a}, {b}, {c};
// 1 条指令 = 1024 次 FMA！
```

## 1.2 MMA 硬件架构演变

### 第一代：Volta (SM70)

```
架构：Volta V100
MMA 形状：m8n8k4, m16n8k8
支持数据类型：FP16
FP16 吞吐量：125 TFLOPS
```

### 第二代：Ampere (SM80)

```
架构：Ampere A100
MMA 形状：m16n8k8, m16n8k16, m16n8k32
支持数据类型：FP16, BF16, TF32, FP64
FP16 吞吐量：312 TFLOPS
```

### 第三代：Hopper (SM90)

```
架构：Hopper H100
MMA 形状：GMMA (更大规模)
支持数据类型：FP8, FP16, BF16, FP64, FP128
FP16 吞吐量：989 TFLOPS
```

## 1.3 MMA 指令格式

### 完整格式

```
mma.sync.aligned.m<M>n<N>k<K>.<Dtype>.<Atype>.<Btype>.<Ctype>
    {d0, d1, ...},     // 输出寄存器
    {a0, a1, ...},     // A 矩阵寄存器
    {b0, b1, ...},     // B 矩阵寄存器
    {c0, c1, ...};     // 累加器寄存器
```

### 示例解析

```cuda
// FP16 GEMM: 16×8×8
mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16
    {d0, d1},        // 2 个 32-bit 寄存器 = 4 个 FP16
    {a0, a1},        // 2 个 32-bit 寄存器 = 4 个 FP16
    {b0},            // 1 个 32-bit 寄存器 = 2 个 FP16
    {c0, c1};        // 2 个 32-bit 寄存器 = 4 个 FP16

// FP32 GEMM: 16×8×8 (FP16 输入，FP32 输出)
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    {d0, d1, d2, d3}, // 4 个 FP32 寄存器
    {a0, a1},         // 2 个 32-bit (4 个 FP16)
    {b0},             // 1 个 32-bit (2 个 FP16)
    {c0, c1, c2, c3}; // 4 个 FP32 寄存器
```

## 1.4 MMA 寄存器布局

### 16×8×8 FP16 MMA 寄存器分配

```
A 矩阵 (16×8 FP16 = 128 元素):
┌─────────────────────────────────┐
│ a0      │ a1      │ ... │ a7   │  8 个 32-bit 寄存器
│ (2 FP16)│ (2 FP16)│     │      │  = 16 FP16 per register
└─────────────────────────────────┘

B 矩阵 (8×8 FP16 = 64 元素):
┌─────────────────────────────────┐
│ b0      │ b1      │ b2    │ b3 │  4 个 32-bit 寄存器
└─────────────────────────────────┘

C/D 矩阵 (16×8 FP16 = 128 元素):
┌─────────────────────────────────┐
│ d0      │ d1      │ ... │ d7   │  8 个 32-bit 寄存器
└─────────────────────────────────┘
```

### 线程到寄存器的映射

```
32 个线程参与 1 个 16×8×8 MMA 操作：

Thread 0-1:  负责输出 D[0:1, 0:1]
Thread 2-3:  负责输出 D[0:1, 2:3]
...
Thread 30-31:负责输出 D[0:1, 6:7]

每个线程负责 2×2 的输出块
```

## 1.5 MMA 与 Tensor Core 的关系

```
┌─────────────────────────────────────────────────┐
│            NVIDIA 矩阵加速技术                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  市场名称：Tensor Core                           │
│  编程接口：MMA (Matrix Multiply-Accumulate)     │
│  PTX 指令：mma.sync...                          │
│                                                 │
│  关系：Tensor Core = MMA 硬件单元                │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 1.6 CUTE MMA API

### 基础用法

```cpp
#include <cute/atom/mma_traits.hpp>

using namespace cute;

// 选择 MMA 操作
using MMA_Op = MMA_Traits<SM80_16x8x8_F32F16F16F32_TN>;

// 创建 MMA Atom
auto mma_atom = make_mma_atom(MMA_Op{});

// 准备数据
auto A = make_tensor(smem_a, layout_a);
auto B = make_tensor(smem_b, layout_b);
auto C = make_tensor(reg_c, layout_c);

// 执行 MMA
mma(mma_atom, A, B, C);
```

### MMA Traits 参数

```cpp
// SM80_16x8x8_F32F16F16F32_TN 解析：
// SM80:       Ampere 架构
// 16x8x8:     M=16, N=8, K=8
// F32:        输出类型 (D)
// F16:        A 矩阵类型
// F16:        B 矩阵类型
// F32:        累加器类型 (C)
// TN:         Transpose A, No transpose B
```

## 1.7 MMA 性能优势

### 理论吞吐量对比

```
A100 GPU (SM80):

CUDA Core FP16:
  - 108 SM × 64 FP16 Core/SM × 2 FMA/cycle × 1.4 GHz
  = ~19 TFLOPS

Tensor Core FP16:
  - 108 SM × 4 TC/SM × 1024 FMA/cycle × 1.4 GHz
  = ~624 TFLOPS (稀疏)
  = ~312 TFLOPS (稠密)

性能提升：16-32 倍！
```

### 实际 GEMM 性能

```
1024×1024×1024 GEMM (FP16):

实现方式          时间 (ms)    TFLOPS
─────────────────────────────────────
朴素 CUDA         5.0         0.7
共享内存优化      1.5         2.3
MMA 指令          0.3         11.5
CUTLASS          0.25        13.8
```

## 1.8 小结

| 知识点 | 要点 |
|--------|------|
| MMA 定义 | 矩阵乘累加硬件加速 |
| 性能优势 | 16-32 倍于 CUDA Core |
| 指令格式 | m<M>n<N>k<K>.<type> |
| 支持架构 | SM70+ (Volta+) |
| CUTE API | MMA_Traits<...> |

**下一步**: 阅读 [02_MMA 硬件架构.md](02_MMA 硬件架构.md)
