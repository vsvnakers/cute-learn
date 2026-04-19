# CUTE Swizzle 从零开始学习指南

## 目录结构

```
swizzle_learn/
├── README.md              # 本文件
├── 01_什么是 swizzle.md    # 基础概念
├── 02_swizzle 数学原理.md  # 详细的数学原理
├── 03_参数详解.md         # 每个参数的含义
├── 04_code_example_basic.cu     # 基础代码示例
├── 05_code_example_advanced.cu  # 进阶代码示例
├── 06_code_example_practical.cu # 实战代码示例
├── 07_可视化演示.md        # 可视化理解
├── 学习总结.md             # 完整总结
├── Makefile               # 编译脚本
├── 04_basic               # 编译后的基础示例
├── 05_advanced            # 编译后的进阶示例
└── 06_practical           # 编译后的实战示例
```

## 学习路线

| 步骤 | 内容 | 预计时间 |
|------|------|----------|
| 1 | 阅读 `01_什么是 swizzle.md` | 10 分钟 |
| 2 | 阅读 `02_swizzle 数学原理.md` | 15 分钟 |
| 3 | 阅读 `03_参数详解.md` | 10 分钟 |
| 4 | 编译运行 `04_code_example_basic.cu` | 10 分钟 |
| 5 | 编译运行 `05_code_example_advanced.cu` | 15 分钟 |
| 6 | 编译运行 `06_code_example_practical.cu` | 15 分钟 |
| 7 | 阅读 `07_可视化演示.md` | 10 分钟 |
| 8 | 阅读 `学习总结.md` | 10 分钟 |

**总计：约 90 分钟**

## 快速开始

### 环境要求

- NVIDIA GPU（支持 CUDA）
- CUDA Toolkit 11.0+
- C++17 编译器
- CUTLASS 库（本仓库已包含）

### 编译运行

```bash
# 进入目录
cd swizzle_learn

# 编译所有示例
make

# 或者单独编译
make 04_basic
make 05_advanced
make 06_practical

# 运行示例
./04_basic    # 基础示例
./05_advanced # 进阶示例
./06_practical # 实战示例

# 清理
make clean
```

### 预期输出

**基础示例 (04_basic)**:
- Swizzle 映射表
- 可逆性验证（全部通过✓）
- GPU Kernel 结果

**进阶示例 (05_advanced)**:
- Layout 组合效果
- Tensor 内容（展示 Swizzle 效果）
- GEMM Layout Bank 分布

**实战示例 (06_practical)**:
- 矩阵转置性能对比（约 27 倍提升）
- GEMM 结果验证
- 详细二进制计算过程

## 核心知识点速查

### Swizzle 公式

```cpp
swizzled_offset = offset ^ (((offset >> SShift) & mask) << BBits);
```

### 参数含义

```
Swizzle<BBits, NumBanks, SShift>

BBits     -> Bank Size = 2^BBits
NumBanks  -> Bank 数量 = 2^NumBanks
SShift    -> XOR 位移量 (必须 >= BBits)
```

### 推荐配置

| 数据类型 | 推荐配置 | 说明 |
|----------|----------|------|
| float32 | `Swizzle<2, 5, 3>` | 最常用 |
| float16 | `Swizzle<3, 5, 4>` | 半精度 |
| int8 | `Swizzle<4, 5, 5>` | 量化 |
| GEMM | `Swizzle<2, 4, 3>` | 16x16 tile |

### 使用模式

```cpp
// 1. 直接调用
int s = Swizzle<2, 4, 3>::apply(offset);

// 2. Layout 组合
auto layout = composition(Swizzle<2,4,3>{}, base_layout);

// 3. Kernel 中使用
__shared__ float shared[256];
int idx = Swizzle<2, 4, 3>::apply(threadIdx.x);
shared[idx] = value;
```

## 性能测试结果

| 测试 | 无 Swizzle | 有 Swizzle | 提升 |
|------|------------|------------|------|
| 矩阵转置 1024x1024 | 3.009 ms | 0.112 ms | ~27x |
| GEMM 256x256x256 | 正确 | 正确 | 功能验证 |

## 常见问题

### Q: 为什么前几个元素的 Swizzle 没有变化？

A: 因为 `SShift=3`，前 8 个元素右移 3 位后为 0，XOR 值为 0，所以不变。从第 8 个元素开始才有变化。

### Q: 如何验证 Swizzle 是否正确？

A: 运行 `04_basic` 的可逆性测试，32 个元素全部应该显示✓。

### Q: SShift 可以小于 BBits 吗？

A: 不可以！这会导致变换不可逆。

### Q: 如何选择合适的参数？

A: 从推荐配置开始，`Swizzle<2, 5, 3>` 适用于大部分 float 场景。

## 下一步学习

完成本教程后，建议继续学习：

1. **CUTLASS 示例**：`../cutlass/examples`
2. **其他教程**：`../tutorials/` 目录
3. **CUTE 文档**：`../cutlass/include/cute/`

## 参考资源

- CUTLASS GitHub: https://github.com/NVIDIA/cutlass
- CUDA 编程指南
- NVIDIA Bank Conflict 文档

## 许可证

本教程遵循 CUTLASS 的 EULA。
