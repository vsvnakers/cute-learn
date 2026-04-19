# CUTE Swizzle 从零开始学习指南

## 目录结构

```
swizzle_learn/
├── README.md                    # 本文件
├── 01_什么是 swizzle.md          # 基础概念
├── 02_swizzle 数学原理.md        # 详细的数学原理
├── 03_参数详解.md               # 每个参数的含义
├── 04_code_example_basic.cu     # 基础代码示例
├── 05_code_example_advanced.cu  # 进阶代码示例
├── 06_code_example_practical.cu # 实战代码示例
├── 07_可视化演示.md              # 可视化理解
├── 学习总结.md                   # 完整总结
├── Makefile                     # 编译脚本
├── tma/                         # TMA 专题教程
│   ├── README.md                # TMA 教程入口
│   ├── 01_TMA 基础概念.md        # TMA 硬件单元
│   ├── 02_TMA_Swizzle 参数详解.md # TMA Swizzle 配置
│   ├── 03_TMA_vs 常规 Swizzle.md # 对比分析
│   ├── 04_code_tma_basic.cu     # TMA 基础代码
│   ├── 05_code_tma_advanced.cu  # TMA 进阶代码
│   ├── 06_code_tma_gemm.cu      # TMA GEMM 实战
│   ├── 07_TMA 可视化.md          # TMA 可视化
│   └── 08_TMA 学习总结.md        # TMA 总结
├── 04_basic                     # 编译后的基础示例
├── 05_advanced                  # 编译后的进阶示例
└── 06_practical                 # 编译后的实战示例
```

## 学习路线

### 第一部分：常规 Swizzle (Ampere+)

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

**常规部分总计：约 90 分钟**

### 第二部分：TMA Swizzle (Hopper+)

| 步骤 | 内容 | 预计时间 |
|------|------|----------|
| 1 | 阅读 `tma/README.md` | 5 分钟 |
| 2 | 阅读 `tma/01_TMA 基础概念.md` | 15 分钟 |
| 3 | 阅读 `tma/02_TMA_Swizzle 参数详解.md` | 15 分钟 |
| 4 | 阅读 `tma/03_TMA_vs 常规 Swizzle.md` | 10 分钟 |
| 5 | 编译运行 `tma/04_code_tma_basic.cu` | 10 分钟 |
| 6 | 编译运行 `tma/05_code_tma_advanced.cu` | 15 分钟 |
| 7 | 编译运行 `tma/06_code_tma_gemm.cu` | 15 分钟 |
| 8 | 阅读 `tma/07_TMA 可视化.md` | 15 分钟 |
| 9 | 阅读 `tma/08_TMA 学习总结.md` | 10 分钟 |

**TMA 部分总计：约 100 分钟**

**全部总计：约 190 分钟（3 小时 10 分钟）**

## 快速开始

### 环境要求

- NVIDIA GPU（支持 CUDA）
- CUDA Toolkit 11.0+
- C++17 编译器
- CUTLASS 库（本仓库已包含）

**注意**: TMA 示例需要 **Hopper 架构** (H100, H200) 或更新 GPU

### 编译运行

```bash
# 进入目录
cd swizzle_learn

# 编译所有常规示例
make

# 编译所有 TMA 示例 (需要 sm_90 GPU)
make tma_all

# 或者单独编译
make 04_basic
make 05_advanced
make 06_practical
make 04_tma_basic
make 05_tma_advanced
make 06_tma_gemm

# 运行示例
./04_basic       # 常规基础示例
./05_advanced    # 常规进阶示例
./06_practical   # 常规实战示例

# 运行 TMA 示例 (需要 Hopper GPU)
./04_tma_basic
./05_tma_advanced
./06_tma_gemm

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
| TMA | `Swizzle<2, 5, 3>` | Hopper 默认 |

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

// 4. TMA 方式 (Hopper)
auto tma_copy = make_tma_copy(gmem_tensor, smem_layout,
                               Swizzle<2, 5, 3>{}, 1);
copy(tma_copy, coord, smem_tensor);
```

## 性能测试结果

| 测试 | 无 Swizzle | 有 Swizzle | TMA Swizzle | 提升 |
|------|------------|------------|-------------|------|
| 矩阵转置 1024x1024 | 3.009 ms | 0.112 ms | 0.045 ms | ~67x |
| GEMM 256x256x256 | 正确 | 正确 | 正确 | 功能验证 |

## 常见问题

### Q: 为什么前几个元素的 Swizzle 没有变化？

A: 因为 `SShift=3`，前 8 个元素右移 3 位后为 0，XOR 值为 0，所以不变。从第 8 个元素开始才有变化。

### Q: 如何验证 Swizzle 是否正确？

A: 运行 `04_basic` 的可逆性测试，32 个元素全部应该显示✓。

### Q: SShift 可以小于 BBits 吗？

A: 不可以！这会导致变换不可逆。

### Q: 如何选择合适的参数？

A: 从推荐配置开始，`Swizzle<2, 5, 3>` 适用于大部分 float 场景。

### Q: TMA 示例无法编译怎么办？

A: TMA 需要 Hopper 架构 (sm_90)。如果没有 Hopper GPU，只能学习理论，无法实际运行。

### Q: TMA 相比软件 Swizzle 有多大提升？

A: 通常 2-3 倍，在某些场景下（如 GEMM）可达更高。

## 学习路径建议

```
常规 Swizzle 学习
       ↓
   理解 Bank Conflict
       ↓
   掌握 XOR 变换
       ↓
   学会参数选择
       ↓
   代码实践
       ↓
       ├──────→ TMA 学习 (如果有 Hopper GPU)
       │           ↓
       │       理解 TMA 硬件
       │           ↓
       │       学习描述符
       │           ↓
       │       掌握 mbarrier
       │           ↓
       │       GEMM 实战
       │
       └──────→ CUTLASS 学习 (如果没有 Hopper GPU)
                   ↓
               阅读源码
                   ↓
               理解应用
```

## 下一步学习

完成本教程后，建议继续学习：

1. **CUTLASS 示例**: `../cutlass/examples`
2. **其他教程**: `../tutorials/` 目录
3. **CUTE 文档**: `../cutlass/include/cute/`
4. **TMA 专题**: `tma/` 目录

## 参考资源

- CUTLASS GitHub: https://github.com/NVIDIA/cutlass
- CUDA 编程指南
- NVIDIA Bank Conflict 文档
- Hopper 架构白皮书

## 许可证

本教程遵循 CUTLASS 的 EULA。
