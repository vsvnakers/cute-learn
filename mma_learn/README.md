# CUTE MMA 从零开始学习指南

## 目录结构

```
mma_learn/
├── README.md                    # 本文件
├── 01_MMA 基础概念.md            # MMA 硬件单元基础
├── 02_MMA 硬件架构.md            # 各代 MMA 硬件架构
├── 03_MMA 指令详解.md            # MMA 指令格式和参数
├── 04_code_mma_basic.cu         # 基础 MMA 代码示例
├── 05_code_mma_advanced.cu      # 进阶 MMA 代码示例
├── 06_code_mma_gemm.cu          # MMA GEMM 实战
├── 07_MMA 可视化.md              # 可视化演示
├── 08_MMA 学习总结.md            # 完整总结
├── Makefile                     # 编译脚本
├── 04_mma_basic                 # 编译后的基础示例
├── 05_mma_advanced              # 编译后的进阶示例
└── 06_mma_gemm                  # 编译后的实战示例
```

## 学习路线

| 步骤 | 内容 | 预计时间 |
|------|------|----------|
| 1 | 阅读 `01_MMA 基础概念.md` | 15 分钟 |
| 2 | 阅读 `02_MMA 硬件架构.md` | 20 分钟 |
| 3 | 阅读 `03_MMA 指令详解.md` | 20 分钟 |
| 4 | 编译运行 `04_code_mma_basic.cu` | 15 分钟 |
| 5 | 编译运行 `05_code_mma_advanced.cu` | 20 分钟 |
| 6 | 编译运行 `06_code_mma_gemm.cu` | 20 分钟 |
| 7 | 阅读 `07_MMA 可视化.md` | 15 分钟 |
| 8 | 阅读 `08_MMA 学习总结.md` | 10 分钟 |

**总计：约 135 分钟**

## 快速开始

### 环境要求

- NVIDIA GPU（Compute Capability 7.0+）
- CUDA Toolkit 11.0+
- C++17 编译器
- CUTLASS 库（本仓库已包含）

### 编译运行

```bash
# 进入目录
cd mma_learn

# 编译所有示例
make

# 或者单独编译
make 04_mma_basic
make 05_mma_advanced
make 06_mma_gemm

# 运行示例
./04_mma_basic     # 基础示例
./05_mma_advanced  # 进阶示例
./06_mma_gemm      # 实战示例

# 清理
make clean
```

## 什么是 MMA？

**MMA (Matrix Multiply-Accumulate)** 是 NVIDIA GPU 的矩阵乘累加硬件单元。

### MMA vs CUDA Core

| 特性 | CUDA Core | Tensor Core (MMA) |
|------|-----------|-------------------|
| 计算类型 | 标量/向量 | 矩阵×矩阵 |
| 吞吐量 | 1 FMA/cycle | 1024+ FMA/cycle |
| 精度 | FP32/FP64 | FP16/TF32/BF16/INT8 |
| 适用场景 | 通用计算 | 深度学习/GEMM |

### 各代 MMA 性能对比

| 架构 | Tensor Core | FP16 吞吐量 |
|------|-------------|-------------|
| Volta (V100) | 1st Gen | 125 TFLOPS |
| Ampere (A100) | 2nd Gen | 312 TFLOPS |
| Hopper (H100) | 3rd Gen | 989 TFLOPS |

## MMA 指令格式

```cuda
// Ampere SM80 MMA 指令示例
mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16
    {d0, d1},        // 输出 D (2x 32-bit)
    {a0, a1},        // 输入 A (2x 32-bit)
    {b0},            // 输入 B (1x 32-bit)
    {c0, c1};        // 累加器 C (2x 32-bit)
```

### 参数含义

```
m16n8k8: M=16, N=8, K=8 的矩阵乘法
row.col: A 是 row-major, B 是 col-major
f16.f16.f16.f16: D=A=B=C=FP16
```

## 核心知识点速查

### MMA 形状

| 架构 | MMA 形状 |
|------|----------|
| SM70/75 | m8n8k4, m16n8k8 |
| SM80 | m16n8k8, m16n8k16, m16n8k32 |
| SM89 | m16n8k32 (FP8) |
| SM90 | GMMA (更大形状) |

### 数据类型支持

| 架构 | 支持的数据类型 |
|------|----------------|
| SM70 | FP16 |
| SM75 | FP16, INT8, INT4 |
| SM80 | FP16, BF16, TF32, FP64 |
| SM89 | FP8, FP16, BF16 |
| SM90 | FP8, FP16, BF16, FP64, FP128 |

### CUTE MMA API

```cpp
// CUTE 高级 MMA API
auto mma_op = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>{};

// 创建 Tensor
auto A = make_tensor(smem_a, layout_a);
auto B = make_tensor(smem_b, layout_b);
auto C = make_tensor(reg_c, layout_c);

// 执行 MMA
copy(A, reg_a);
copy(B, reg_b);
mma(mma_op, reg_a, reg_b, reg_c);
```

## 性能对比

| 实现方式 | GEMM 性能 (A100 FP16) |
|----------|----------------------|
| 朴素 CUDA Core | ~10 TFLOPS |
| 共享内存优化 | ~50 TFLOPS |
| MMA 指令 | ~300 TFLOPS |
| CUTLASS | ~310 TFLOPS |

## 常见问题

### Q: MMA 和 Tensor Core 是什么关系？
A: Tensor Core 是 NVIDIA 的市场名称，MMA 是编程接口名称。

### Q: 我的 GPU 支持 MMA 吗？
A: Compute Capability 7.0+ 支持 MMA（Volta 及更新架构）。

### Q: MMA 支持哪些数据类型？
A: 取决于架构，从 FP16 到 FP8、INT8 等。

## 下一步学习

完成本教程后，建议继续学习：
1. **TMA 教程**: `../swizzle_learn/tma/`
2. **CUTLASS 示例**: `../cutlass/examples/`
3. **Flash Attention**: 理解 MMA 在实际应用中的使用

## 参考资源

- [NVIDIA Tensor Core 文档](https://docs.nvidia.com/cuda/cublas/index.html)
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## 许可证

本教程遵循 CUTLASS 的 EULA。
