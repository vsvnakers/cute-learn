# CUTE & Tensor Core CUDA 编程教程

欢迎！这是一个完整的 CUTE (CUDA Universal Tensor Engine) 学习教程。

## 支持的 GPU

| GPU | 架构 | SM 版本 | 状态 |
|-----|------|---------|------|
| RTX 3060 (笔记本) | Ampere | 86 | ✅ 支持 |
| A100 | Ampere | 80 | ✅ 支持 |
| RTX 4090 | Ada Lovelace | 89 | ✅ 支持 |

## 目录

```
tutorials/
├── README.md              # 本文件
├── GUIDE.md               # 详细学习指南
├── CHEATSHEET.md          # 快速参考手册
├── 跨平台编译.md          # 编译配置说明
├── build.sh               # 自动化编译脚本
├── CMakeLists.txt         # 编译配置
│
├── 01_layout/             # 第一课：Layout 布局系统
├── 02_swizzle/            # 第二课：Swizzle 技术
├── 03_bank_conflict/      # 第三课：Bank Conflict 解决
├── 04_mma/                # 第四课：MMA 指令
├── 05_gemm/               # 第五课：完整 GEMM 实现
└── 06_attention/          # 第六课：Flash Attention
```

## 快速开始

### 方法 1：自动编译脚本（推荐）

```bash
cd tutorials

# 自动检测当前 GPU 并编译
./build.sh

# 或者指定目标
./build.sh laptop      # RTX 3060 笔记本
./build.sh server      # A100 服务器
./build.sh universal   # 两者都支持
```

### 方法 2：CMake 手动编译

```bash
cd tutorials
mkdir build && cd build

# 通用编译（支持 RTX 3060 + A100）
cmake ..
make -j8

# 或者指定架构
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..  # 仅 RTX 3060
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..  # 仅 A100
```

### 运行

```bash
./01_layout_basic
./02_swizzle_basic
./03_bank_conflict
./04_mma_basic
./05_gemm
./06_flash_attention
```

## 课程概览

| 课程 | 主题 | 核心概念 | 预计时间 |
|------|------|----------|----------|
| 01 | Layout 布局系统 | Shape, Stride, Tensor | 1 小时 |
| 02 | Swizzle 技术 | XOR 地址变换 | 1 小时 |
| 03 | Bank Conflict | 共享内存优化 | 2 小时 |
| 04 | MMA 指令 | Tensor Core | 2 小时 |
| 05 | GEMM 实现 | 完整矩阵乘法 | 3 小时 |
| 06 | Flash Attention | Attention 优化 | 4 小时 |

## 学习目标

完成本教程后，你将能够：

1. **理解 CUTE Layout** - 掌握多维数组索引抽象
2. **优化内存访问** - 使用 Swizzle 避免 Bank Conflict
3. **使用 Tensor Core** - 调用 MMA 指令加速计算
4. **实现 GEMM** - 从零编写优化的矩阵乘法
5. **实现 Attention** - 编写高效的 Flash Attention

## 前置要求

- 熟悉 C++ 编程
- 了解 CUDA 基础
- 有 GPU 编程经验

## 硬件要求

- **最低**：支持 CUDA 的 NVIDIA GPU
- **推荐**：RTX 30 系列或更新（Ampere 架构）
- **编译**：CUDA 11.0+

## 学习资源

- [详细指南](GUIDE.md) - 完整的学习路线和说明
- [快速参考](CHEATSHEET.md) - API 和代码片段
- [CUTLASS 文档](https://nvidia.github.io/cutlass/)

## 课程详情

### 第一课：Layout 布局系统

学习 CUTE 的核心概念：

```cpp
#include <cute/layout.hpp>

// 创建 Layout
auto layout = make_layout(make_shape(8, 8));

// 创建 Tensor
auto tensor = make_tensor(ptr, layout);

// 访问元素
auto val = tensor(i, j);
```

### 第二课：Swizzle 技术

学习内存地址变换优化：

```cpp
#include <cute/swizzle.hpp>

// 创建 Swizzle
auto swz = swizzle<2, 4, 2>{};

// 创建 Swizzle Layout
auto swizzled = make_swizzle_layout(layout, swz);
```

### 第三课：Bank Conflict 解决

理解并解决共享内存冲突：

```cpp
// 使用 padding 避免 Bank Conflict
__shared__ float smem[32 * 33];  // 33 = 32 + 1

// 使用 Swizzle
auto smem_layout = make_swizzle_layout(
    make_layout(make_shape(32, 32)),
    swizzle<2, 4, 2>{}
);
```

### 第四课：MMA 指令

掌握 Tensor Core 编程：

```cpp
#include <cute/atom/mma_atom.hpp>

using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

// MMA 操作：D = A * B + C
```

### 第五课：完整 GEMM 实现

实现优化的矩阵乘法：

```cpp
template<int BLOCK_SIZE>
__global__ void tiled_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    // 共享内存分块
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    // 分块计算
    ...
}
```

### 第六课：Flash Attention

实现高效的 Attention 机制：

```cpp
// Attention(Q, K, V) = softmax(QK^T/√d) V
template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void flash_attention_simple(
    const float* Q, const float* K, const float* V,
    float* O,
    int seq_len, int num_heads) {
    // 在线 softmax
    // 分块计算
    ...
}
```

## 常见问题

**Q: 编译错误找不到头文件？**

确保添加了 CUTLASS 路径：
```bash
nvcc -I../cutlass/include your_code.cu
```

**Q: 运行时 CUDA 错误？**

检查 GPU 架构兼容性：
```bash
nvcc -arch=sm_80 ...  # Ampere
nvcc -arch=sm_90 ...  # Hopper
```

**Q: 如何调试？**

使用 `cudaGetErrorString()` 检查错误：
```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
}
```

## 下一步

完成教程后：

1. 阅读 [CUTLASS 源码](https://github.com/NVIDIA/cutlass)
2. 学习 [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
3. 尝试实现其他算子（Conv, LayerNorm, etc.）

祝你学习顺利！
