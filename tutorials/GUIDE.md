# CUTE & Tensor Core CUDA 编程 - 完整学习指南

## 课程结构

本教程共 6 课，从基础到高级，带你掌握 CUTE 编程：

```
tutorials/
├── 01_layout/           # 第一课：Layout 布局系统
│   ├── README.md        # 概念讲解
│   └── 01_layout_basic.cu
├── 02_swizzle/          # 第二课：Swizzle 技术
│   ├── README.md
│   └── 02_swizzle_basic.cu
├── 03_bank_conflict/    # 第三课：Bank Conflict 解决
│   ├── README.md
│   └── 03_bank_conflict.cu
├── 04_mma/              # 第四课：MMA 指令
│   ├── README.md
│   └── 04_mma_basic.cu
├── 05_gemm/             # 第五课：完整 GEMM 实现
│   ├── README.md
│   └── 05_gemm.cu
└── 06_attention/        # 第六课：Flash Attention
    ├── README.md
    └── 06_flash_attention.cu
```

## 编译方法

### 方法 1：使用 CMake（推荐）

```bash
cd /mnt/d/cute-learn/tutorials
mkdir build && cd build
cmake ..
make -j8
```

### 方法 2：手动编译每个示例

```bash
# 第一课
nvcc -std=c++17 -arch=sm_80 -I../cutlass/include 01_layout/01_layout_basic.cu -o 01_layout_basic

# 第二课
nvcc -std=c++17 -arch=sm_80 -I../cutlass/include 02_swizzle/02_swizzle_basic.cu -o 02_swizzle_basic

# 第三课
nvcc -std=c++17 -arch=sm_80 -I../cutlass/include 03_bank_conflict/03_bank_conflict.cu -o 03_bank_conflict

# 第四课
nvcc -std=c++17 -arch=sm_80 -I../cutlass/include 04_mma/04_mma_basic.cu -o 04_mma_basic

# 第五课
nvcc -std=c++17 -arch=sm_80 -I../cutlass/include 05_gemm/05_gemm.cu -o 05_gemm

# 第六课
nvcc -std=c++17 -arch=sm_80 -I../cutlass/include 06_attention/06_flash_attention.cu -o 06_flash_attention
```

## 运行测试

```bash
# 运行所有课程
./01_layout_basic
./02_swizzle_basic
./03_bank_conflict
./04_mma_basic
./05_gemm
./06_flash_attention
```

## 学习路线

### 第一阶段：基础（第 1-3 课）

1. **Layout 布局系统**
   - 理解 Shape 和 Stride
   - Layout 的数学定义
   - Tensor 的创建和访问

2. **Swizzle 技术**
   - Swizzle 函数原理
   - Swizzle Layout
   - 实际应用

3. **Bank Conflict 解决**
   - 理解 Bank Conflict
   - Padding 技术
   - Swizzle 优化

### 第二阶段：核心（第 4-5 课）

4. **MMA 指令**
   - Tensor Core 架构
   - MMA 指令格式
   - CUTE MMA Atom

5. **完整 GEMM**
   - 朴素实现
   - 共享内存分块
   - Tensor Core 加速

### 第三阶段：实战（第 6 课）

6. **Flash Attention**
   - Attention 数学
   - Flash Attention 原理
   - CUTE 实现

## 硬件要求

- **最低要求**：支持 CUDA 的 NVIDIA GPU
- **Tensor Core**：需要 Volta 或更新架构（SM 70+）
- **推荐**：Ampere (RTX 30 系列) 或更新

### 架构兼容性

| 示例 | 最低架构 | 推荐架构 |
|------|----------|----------|
| 01-03 | SM 50 | SM 80 |
| 04-05 | SM 70 | SM 80 |
| 06 | SM 70 | SM 80+ |

修改架构编译：
```bash
nvcc -std=c++17 -arch=sm_90 ...  # Hopper
nvcc -std=c++17 -arch=sm_80 ...  # Ampere
nvcc -std=c++17 -arch=sm_75 ...  # Turing
```

## 调试技巧

### 1. 检查错误

```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
```

### 2. 使用 cuda-gdb

```bash
cuda-gdb ./your_program
(gdb) cuda kernels
(gdb) cuda launch your_kernel
```

### 3. 性能分析

```bash
# 使用 Nsight Systems
nsys profile ./your_program

# 使用 Nsight Compute
ncu ./your_program
```

## 进阶资源

- [CUTLASS 官方文档](https://nvidia.github.io/cutlass/)
- [CUTE 论文](https://arxiv.org/abs/2305.12902)
- [NVIDIA CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)

## 常见问题

### Q: 编译错误 "identifier "make_layout" is undefined"
A: 确保添加了 CUTLASS include 路径：`-I/path/to/cutlass/include`

### Q: 运行时错误 "CUDA ERROR: unknown error"
A: 检查你的 GPU 架构是否支持指定的 SM 版本

### Q: 性能不如预期
A: 使用 Nsight 分析，检查 Bank Conflict 和内存访问模式

祝你学习顺利！
