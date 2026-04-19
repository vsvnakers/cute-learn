# PTX 汇编从零开始学习指南

## 目录结构

```
ptx_learn/
├── README.md                    # 本文件
├── 01_PTX 基础概念.md            # PTX 是什么，为什么学习
├── 02_PTX 数据类型.md            # PTX 数据类型和表示
├── 03_PTX 指令集.md              # 常用 PTX 指令详解
├── 04_code_ptx_basic.cu         # 基础 PTX 示例
├── 05_code_ptx_advanced.cu      # 进阶 PTX 示例
├── 06_code_ptx_intrinsics.cu    # 内联汇编实战
├── 07_PTX 调试技巧.md            # PTX 调试和优化工具
├── 08_PTX 学习总结.md            # 完整总结
├── Makefile                     # 编译脚本
├── 04_ptx_basic.txt             # 基础示例输出
├── 05_ptx_advanced.txt          # 进阶示例输出
└── 06_ptx_intrinsics.txt        # 内联汇编输出
```

## 学习路线

| 步骤 | 内容 | 预计时间 |
|------|------|----------|
| 1 | 阅读 `01_PTX 基础概念.md` | 20 分钟 |
| 2 | 阅读 `02_PTX 数据类型.md` | 20 分钟 |
| 3 | 阅读 `03_PTX 指令集.md` | 30 分钟 |
| 4 | 编译运行 `04_code_ptx_basic.cu` | 20 分钟 |
| 5 | 编译运行 `05_code_ptx_advanced.cu` | 25 分钟 |
| 6 | 编译运行 `06_code_ptx_intrinsics.cu` | 25 分钟 |
| 7 | 阅读 `07_PTX 调试技巧.md` | 20 分钟 |
| 8 | 阅读 `08_PTX 学习总结.md` | 15 分钟 |

**总计：约 175 分钟**

## 快速开始

### 环境要求

- NVIDIA GPU（任意 Compute Capability）
- CUDA Toolkit 11.0+
- C++17 编译器
- nvcc 编译器

### 编译运行

```bash
# 进入目录
cd ptx_learn

# 编译所有示例
make

# 或者单独编译
make 04_ptx_basic
make 05_ptx_advanced
make 06_ptx_intrinsics

# 运行示例
./04_ptx_basic     # 基础示例
./05_ptx_advanced  # 进阶示例
./06_ptx_intrinsics  # 内联汇编示例

# 保存输出到 txt 文件
make save_output

# 清理
make clean
```

## 什么是 PTX？

**PTX (Parallel Thread Execution)** 是 NVIDIA CUDA 的虚拟汇编语言和指令集架构。

### PTX 在编译流程中的位置

```
CUDA C++ 代码 (.cu)
       ↓
   nvcc 前端
       ↓
   PTX 代码 (.ptx)     ← 虚拟 ISA，与硬件无关
       ↓
   ptxas (PTX 汇编器)
       ↓
   SASS 代码 (.cubin)  ← 真实 GPU 机器码
       ↓
   加载到 GPU 执行
```

### 为什么学习 PTX？

| 原因 | 说明 |
|------|------|
| 理解底层执行 | 看清 CUDA 代码如何映射到硬件 |
| 性能优化 | 识别瓶颈，优化关键代码路径 |
| 调试工具 | 使用 `cuobjdump --dump-sass` 分析 |
| 新特性学习 | 第一时间了解新指令和硬件特性 |
| 内联汇编 | 手写 PTX 实现特殊功能 |

## PTX 代码示例

### 基础 PTX 程序

```ptx
// 简单的向量加法 PTX 代码
.version 6.0
.target sm_50
.address_size 64

.visible .entry vector_add(
    .param .u64 input_a,
    .param .u64 input_b,
    .param .u64 output_c,
    .param .u32 num_elements
) {
    // 读取参数
    ld.param.u64 %rd1, [input_a];
    ld.param.u64 %rd2, [input_b];
    ld.param.u64 %rd3, [output_c];
    ld.param.u32 %r1, [num_elements];

    // 计算全局索引
    cvt.u32.u64 %r2, %ctaid.x;
    cvt.u32.u64 %r3, %ntid.x;
    mul.lo.u32 %r4, %r2, %r3;
    add.u32 %r5, %r4, %tid.x;

    // 边界检查
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra EXIT;

    // 加载数据
    mul.lo.u32 %r6, %r5, 4;
    add.u64 %rd4, %rd1, %r6;
    add.u64 %rd5, %rd2, %r6;
    ld.global.f32 %f1, [%rd4];
    ld.global.f32 %f2, [%rd5];

    // 执行加法
    add.f32 %f3, %f1, %f2;

    // 存储结果
    add.u64 %rd6, %rd3, %r6;
    st.global.f32 [%rd6], %f3;

EXIT:
    ret;
}
```

### C++ 内联 PTX 示例

```cpp
__global__ void vector_add_inline(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val_a, val_b, val_c;

        // 内联 PTX 加载
        asm volatile("ld.global.f32 %0, [%1];"
                     : "=f"(val_a) : "l"(a + idx));
        asm volatile("ld.global.f32 %0, [%1];"
                     : "=f"(val_b) : "l"(b + idx));

        // 执行加法
        val_c = val_a + val_b;

        // 内联 PTX 存储
        asm volatile("st.global.f32 [%0], %1;"
                     : : "l"(c + idx), "f"(val_c));
    }
}
```

## PTX 指令分类

### 数据移动指令

| 指令 | 说明 | 示例 |
|------|------|------|
| `ld` | 从内存加载 | `ld.global.f32 %f1, [%rd1];` |
| `st` | 存储到内存 | `st.global.f32 [%rd1], %f1;` |
| `mov` | 寄存器间移动 | `mov.b32 %r1, %r2;` |
| `cvt` | 类型转换 | `cvt.u32.f32 %r1, %f1;` |

### 算术指令

| 指令 | 说明 | 示例 |
|------|------|------|
| `add` | 加法 | `add.f32 %f3, %f1, %f2;` |
| `sub` | 减法 | `sub.f32 %f3, %f1, %f2;` |
| `mul` | 乘法 | `mul.f32 %f3, %f1, %f2;` |
| `fma` | 乘加 | `fma.rn.f32 %f3, %f1, %f2, %f4;` |
| `div` | 除法 | `div.f32 %f3, %f1, %f2;` |

### 比较指令

| 指令 | 说明 | 示例 |
|------|------|------|
| `setp` | 设置谓词 | `setp.eq.f32 %p1, %f1, %f2;` |
| `set` | 设置寄存器 | `set.eq.u32 %r1, %r2, 0;` |

### 控制流指令

| 指令 | 说明 | 示例 |
|------|------|------|
| `bra` | 无条件分支 | `bra LOOP_BEGIN;` |
| `@p bra` | 条件分支 | `@%p1 bra EXIT;` |
| `call` | 函数调用 | `call func_name();` |
| `ret` | 返回 | `ret;` |

### 特殊寄存器

| 寄存器 | 说明 |
|--------|------|
| `%tid.x/y/z` | 线程索引 |
| `%ntid.x/y/z` | 块内线程数 |
| `%ctaid.x/y/z` | 块索引 |
| `%nctaid.x/y/z` | 网格内块数 |
| `%clock` | 时钟周期计数器 |
| `%smid` | SM ID |

## 性能对比

| 实现方式 | 相对性能 |
|----------|---------|
| 朴素 CUDA | 1.0x |
| 优化 CUDA | 2-5x |
| 内联 PTX | 5-10x (特定场景) |
| 手写 SASS | 10-20x (极端优化) |

## 常见问题

### Q: 为什么要学习 PTX？
A: 理解 GPU 底层执行模型，编写高性能代码。

### Q: PTX 和 SASS 有什么区别？
A: PTX 是虚拟 ISA，SASS 是真实 GPU 机器码。

### Q: 如何查看 PTX 代码？
A: 使用 `nvcc -ptx` 编译，或 `cuobjdump --dump-ptx`。

### Q: 内联 PTX 语法是什么？
A: `asm volatile("ptx 指令" : 输出 : 输入 : 内存);`

## 下一步学习

完成本教程后，建议继续学习：
1. **SASS 汇编**: 深入 GPU 机器码层面
2. **CUDA Occupancy Calculator**: 优化资源使用
3. **Nsight Compute**: 性能分析工具

## 参考资源

- [PTX ISA 官方文档](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuobjdump 工具文档](https://docs.nvidia.com/cuda/cuda-binary-utilities/)

## 许可证

本教程遵循 NVIDIA 的 EULA。
