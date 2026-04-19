# 第四课：MMA (Matrix Multiply-Accumulate) 指令

## 1. Tensor Core 架构

### 什么是 Tensor Core？

Tensor Core 是 NVIDIA GPU 中的专用矩阵乘法单元，能够在一个时钟周期内执行完整的矩阵乘法累加运算。

### 各代 Tensor Core 对比

| 架构 | SM 版本 | MMA 尺寸 | 精度支持 | 吞吐量 (FP16) |
|------|---------|----------|----------|---------------|
| Volta | SM 70 | 4x4x4 | FP16 | 8 TFLOPS |
| Turing | SM 75 | 8x8x4 | FP16/INT8 | 16 TFLOPS |
| Ampere | SM 80 | 16x8x8 | FP16/BF16/TF32 | 64 TFLOPS |
| Hopper | SM 90 | 16x8x16+ | FP8/FP16/BF16 | 197 TFLOPS |

## 2. MMA 指令格式

```
D = A × B + C

其中:
- A: M×K 矩阵
- B: K×N 矩阵
- C, D: M×N 矩阵
```

### Ampere SM80 MMA 指令

```cpp
// 16x8x8 FP16 MMA
mma.sync.aligned.m16n8k8.f32.f16.f16.f32
    d,
    a, b, c;
```

## 3. CUTE 中的 MMA 抽象

CUTE 提供了统一的 MMA 接口：
- `mma_atom`：单个 MMA 指令抽象
- `mma_traits`：MMA 配置和属性
- 跨架构支持：SM50/61/70/75/80/89/90
