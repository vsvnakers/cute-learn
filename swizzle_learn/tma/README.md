# Hopper TMA Swizzle 教程

## 目录

1. [TMA 基础概念](01_TMA 基础概念.md)
2. [TMA Swizzle 参数详解](02_TMA_Swizzle 参数详解.md)
3. [TMA 与常规 Swizzle 对比](03_TMA_vs_常规 Swizzle.md)
4. [代码示例 - 基础](04_code_tma_basic.cu)
5. [代码示例 - 进阶](05_code_tma_advanced.cu)
6. [代码示例 - 实战 GEMM](06_code_tma_gemm.cu)
7. [可视化演示](07_TMA 可视化.md)
8. [学习总结](08_TMA 学习总结.md)

## 快速开始

```bash
# 编译所有 TMA 示例
make tma_all

# 或者单独编译
make 04_tma_basic
make 05_tma_advanced
make 06_tma_gemm

# 运行示例
./04_tma_basic
```

## 环境要求

- **GPU 架构**: Hopper (H100, H200) 或更新
- **CUDA**: 12.0+
- **Compute Capability**: sm_90 或更高

**注意**: TMA 是 Hopper 架构的硬件特性，在旧架构上无法运行！

## 什么是 TMA？

**TMA (Tensor Memory Accelerator)** 是 NVIDIA Hopper 架构引入的专用硬件单元，用于：

1. **异步内存拷贝**: 硬件加速的全局内存 ↔ 共享内存拷贝
2. **自动地址计算**: 硬件处理复杂的索引和边界检查
3. **内置 Swizzle 支持**: 硬件级别的 Bank Conflict 避免
4. **多维拷贝**: 原生支持 1D/2D/3D/4D/5D 张量拷贝

### TMA 的优势

| 特性 | 传统方式 | TMA 方式 |
|------|----------|----------|
| 地址计算 | 软件计算 | 硬件计算 |
| 边界检查 | 软件分支 | 硬件处理 |
| Swizzle | 软件 XOR | 硬件内置 |
| 异步性 | mbarrier + 软件 | 硬件 mbarrier |
| 多维索引 | 手动展平 | 原生支持 |

## TMA Swizzle vs 常规 Swizzle

### 常规 Swizzle (Ampere 及更早)

```cpp
// 软件 XOR 操作
int swizzled_idx = Swizzle<2, 4, 3>::apply(linear_idx);
shared_mem[swizzled_idx] = value;
```

### TMA Swizzle (Hopper)

```cpp
// 硬件自动处理 Swizzle
// TMA 描述符中配置 Swizzle 参数
// 拷贝时硬件自动应用
auto tma_copy = make_tma_copy(..., Swizzle<2, 5, 3>{}, ...);
copy(tma_copy, gmem_tensor, smem_tensor);
```

## TMA Swizzle 参数

TMA 使用特殊的 Swizzle 配置，与常规 Swizzle 有所不同：

### TMA 支持的 Swizzle 模式

| Swizzle 配置 | TMA 模式 | 描述 |
|-------------|----------|------|
| `Swizzle<2, 4, 3>` | B128 | 128 位基础 swizzle |
| `Swizzle<2, 5, 2>` | B32  | 32 字节基础 swizzle |
| `Swizzle<2, 5, 3>` | B64  | 64 字节基础 swizzle |
| `Swizzle<2, 6, 3>` | B128 | 128 字节基础 swizzle |

### 关键差异

1. **NumBanks (M 参数)**: TMA 限制为 4, 5, 或 6
2. **BBits (B 参数)**: 通常固定为 2 或 3
3. **SShift (S 参数)**: 有特定组合要求

## 学习路线

```
第一步：阅读 01_TMA 基础概念.md
       └── 理解 TMA 硬件单元
       └── 学习 TMA 描述符
       └── 掌握 mbarrier 机制

第二步：阅读 02_TMA_Swizzle 参数详解.md
       └── TMA Swizzle 的特殊要求
       └── 参数映射关系
       └── 如何选择配置

第三步：阅读 03_TMA_vs 常规 Swizzle.md
       └── 详细对比表
       └── 性能差异
       └── 使用场景

第四步：编译运行代码示例
       └── 04_code_tma_basic.cu (基础)
       └── 05_code_tma_advanced.cu (进阶)
       └── 06_code_tma_gemm.cu (实战)

第五步：阅读可视化和总结
       └── 07_TMA 可视化.md
       └── 08_TMA 学习总结.md
```

## 参考资源

- [NVIDIA Hopper 架构白皮书](https://resources.nvidia.com/en-us-hopper-architecture)
- [CUTLASS 3.x 文档](https://github.com/NVIDIA/cutlass)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## 下一步

开始学习 [01_TMA 基础概念.md](01_TMA 基础概念.md)
