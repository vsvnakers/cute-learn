# 第一课：PTX 基础概念

## 1.1 什么是 PTX？

**PTX (Parallel Thread Execution)** 是 NVIDIA CUDA 平台的虚拟汇编语言和指令集架构。

### PTX 的本质

```
┌─────────────────────────────────────────────────────────────────┐
│                      CUDA 编译流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CUDA C++ 代码 (.cu)                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                           │
│  │   nvcc 前端      │  C++ 语法解析，模板展开，设备代码分离       │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │   PTX 代码       │  ← 虚拟 ISA (Virtual ISA)                 │
│  │   (.ptx)        │     与具体硬件无关                         │
│  └────────┬────────┘     类似 Java 字节码                        │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │   ptxas        │  PTX 汇编器 (PTX Assembler)                │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │   SASS 代码      │  ← 真实 GPU 机器码                         │
│  │   (.cubin)      │     特定 GPU 架构的二进制                  │
│  └────────┬────────┘     类似 x86 机器码                         │
│           │                                                     │
│           ▼                                                     │
│  GPU 硬件执行                                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### PTX vs SASS

| 特性 | PTX | SASS |
|------|-----|------|
| 全称 | Parallel Thread Execution | Streamed Architecture Assembly |
| 性质 | 虚拟 ISA | 真实机器码 |
| 硬件相关 | 否 | 是（特定 GPU 架构） |
| 稳定性 | 稳定（向后兼容） | 每代 GPU 不同 |
| 可读性 | 较高 | 较低 |
| 用途 | 学习、优化、JIT | 深度优化、逆向 |

## 1.2 为什么学习 PTX？

### 理由 1: 理解底层执行模型

```cpp
// CUDA C 代码
c[i] = a[i] + b[i];

// 对应的 PTX 代码
ld.global.f32 %f1, [%rd1];    // 加载 a[i]
ld.global.f32 %f2, [%rd2];    // 加载 b[i]
add.f32 %f3, %f1, %f2;        // 执行加法
st.global.f32 [%rd3], %f3;    // 存储 c[i]
```

通过 PTX，你可以看到：
- 内存访问模式（ld/st）
- 寄存器使用（%f1, %f2, %f3）
- 指令级并行性

### 理由 2: 性能优化

```ptx
// 低效代码：顺序依赖
ld.global.f32 %f1, [%rd1];
mul.f32 %f2, %f1, 2.0;
add.f32 %f3, %f2, 1.0;
st.global.f32 [%rd3], %f3;

// 高效代码：指令级并行
ld.global.f32 %f1, [%rd1];
ld.global.f32 %f4, [%rd4];    // 提前加载下一个数据
mul.f32 %f2, %f1, 2.0;
add.f32 %f3, %f2, 1.0;
st.global.f32 [%rd3], %f3;
```

### 理由 3: 调试和分析

```bash
# 查看 PTX 代码
nvcc -ptx my_kernel.cu -o my_kernel.ptx

# 查看 SASS 代码
cuobjdump --dump-sass my_kernel.cubin

# 查看 PTX 汇编（使用 Nsight Compute）
ncu --dump-config my_kernel
```

### 理由 4: 内联汇编

```cpp
__device__ float fast_sqrt(float x) {
    float result;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}
```

## 1.3 PTX 版本和架构

### PTX 版本历史

| PTX 版本 | CUDA 版本 | 发布年份 | 新特性 |
|---------|----------|---------|--------|
| 1.0 | CUDA 1.0 | 2007 | 基础版本 |
| 2.0 | CUDA 3.0 | 2010 | 双精度支持 |
| 3.0 | CUDA 4.0 | 2011 | GPU 原子操作 |
| 4.0 | CUDA 5.0 | 2012 | 动态并行 |
| 5.0 | CUDA 6.0 | 2013 | 统一内存 |
| 6.0 | CUDA 9.0 | 2017 | Volta 支持 |
| 6.3 | CUDA 10.1 | 2019 | Turing 支持 |
| 7.0 | CUDA 11.0 | 2020 | Ampere 支持 |
| 8.0 | CUDA 12.0 | 2022 | Hopper 支持 |

### 查看 PTX 版本

```ptx
// PTX 文件头部
.version 8.0
.target sm_90
.address_size 64

// 含义:
// .version 8.0  - 使用 PTX ISA 8.0 语法
// .target sm_90 - 目标架构 Hopper (SM90)
// .address_size 64 - 64 位地址空间
```

## 1.4 PTX 程序结构

### 完整 PTX 程序示例

```ptx
// 1. 版本和目标声明
.version 7.0
.target sm_80
.address_size 64

// 2. 可见性声明
.visible .entry vector_add(
    .param .u64 input_a,      // 参数：输入 A 的指针
    .param .u64 input_b,      // 参数：输入 B 的指针
    .param .u64 output_c,     // 参数：输出 C 的指针
    .param .u32 num_elements  // 参数：元素数量
) {
    // 3. 局部变量声明（隐式）
    // PTX 使用虚拟寄存器，由汇编器分配

    // 4. 加载参数
    ld.param.u64 %rd1, [input_a];
    ld.param.u64 %rd2, [input_b];
    ld.param.u64 %rd3, [output_c];
    ld.param.u32 %r1, [num_elements];

    // 5. 计算全局线程索引
    cvt.u32.u64 %r2, %ctaid.x;    // 块索引转 u32
    cvt.u32.u64 %r3, %ntid.x;     // 每块线程数转 u32
    mul.lo.u32 %r4, %r2, %r3;     // 块索引 × 每块线程数
    add.u32 %r5, %r4, %tid.x;     // + 线程索引 = 全局索引

    // 6. 边界检查
    setp.ge.u32 %p1, %r5, %r1;    // 如果索引 >= 数量
    @%p1 bra EXIT;                 // 跳转到退出

    // 7. 计算内存地址
    mul.lo.u32 %r6, %r5, 4;       // 索引 × 4 (float 大小)
    add.u64 %rd4, %rd1, %r6;      // A 的地址
    add.u64 %rd5, %rd2, %r6;      // B 的地址
    add.u64 %rd6, %rd3, %r6;      // C 的地址

    // 8. 加载数据
    ld.global.f32 %f1, [%rd4];    // 加载 A[i]
    ld.global.f32 %f2, [%rd5];    // 加载 B[i]

    // 9. 执行计算
    add.f32 %f3, %f1, %f2;        // C = A + B

    // 10. 存储结果
    st.global.f32 [%rd6], %f3;    // 存储 C[i]

EXIT:
    // 11. 返回
    ret;
}
```

### PTX 代码组成部分

```
┌─────────────────────────────────────────────────────────────────┐
│  PTX 程序结构                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 头部声明                                                    │
│     .version X.Y          PTX ISA 版本                          │
│     .target sm_XX         目标 GPU 架构                          │
│     .address_size 64      地址空间大小                          │
│                                                                 │
│  2. 内存空间声明                                                │
│     .global .var .u32 counter;    全局变量                      │
│     .const .var .u32 config;      常量内存                      │
│                                                                 │
│  3. 函数声明                                                    │
│     .visible .entry kernel(...)   可见 kernel 函数               │
│     .func device_func(...)        设备函数                      │
│                                                                 │
│  4. 参数声明                                                    │
│     .param .u64 ptr             指针参数                        │
│     .param .u32 val             值参数                          │
│                                                                 │
│  5. 函数体                                                      │
│     指令序列                                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 1.5 PTX 内存空间

### 内存空间类型

```
┌─────────────────────────────────────────────────────────────────┐
│  PTX 内存空间                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  .global         全局内存（GPU DRAM）                           │
│    用途：设备内存，主机可访问                                    │
│    延迟：高（400-800 周期）                                      │
│    带宽：高（>900 GB/s）                                        │
│                                                                 │
│  .shared         共享内存（片上 SRAM）                          │
│    用途：线程块内通信                                            │
│    延迟：低（~20 周期）                                          │
│    带宽：极高（TB/s 级）                                         │
│                                                                 │
│  .const          常量内存（只读缓存）                           │
│    用途：只读常量数据                                            │
│    延迟：中（缓存命中~2 周期）                                   │
│    带宽：中                                                    │
│                                                                 │
│  .local          本地内存（溢出到全局）                         │
│    用途：寄存器溢出，大局部变量                                  │
│    延迟：高（同 global）                                        │
│                                                                 │
│  .param          参数内存（内核参数）                           │
│    用途：传递内核参数                                            │
│    延迟：低                                                    │
│                                                                 │
│  .reg            寄存器（最快）                                 │
│    用途：临时变量，计算结果                                      │
│    延迟：极低（1 周期）                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 内存空间使用示例

```ptx
// 全局内存
.global .align 4 .u32 global_counter;

// 常量内存
.const .align 4 .u32 config_value;

// 共享内存（在 kernel 中声明）
shared .align 4 .u32 shared_data[256];

// 本地内存（自动变量，可能溢出）
.entry my_kernel() {
    .reg .u32 local_var;  // 可能在寄存器或 local 内存
}

// 参数内存
.entry my_kernel(.param .u64 input_ptr) {
    ld.param.u64 %rd1, [input_ptr];
}
```

## 1.6 PTX 寄存器

### 寄存器命名规则

```
PTX 寄存器命名:
%前缀 + 类型标识 + 数字

类型标识:
.r 或无  - 通用寄存器
.b      - 位类型
.u      - 无符号整数
.s      - 有符号整数
.f      - 浮点数
.v      - 向量
.p      - 谓词（布尔）
.rd     - 64 位地址
```

### 寄存器示例

```ptx
// 通用寄存器
.reg .u32 %r<5>;     // 声明 5 个 u32 寄存器：%r1, %r2, %r3, %r4, %r5
.reg .f32 %f<3>;     // 声明 3 个 f32 寄存器：%f1, %f2, %f3
.reg .u64 %rd<2>;    // 声明 2 个 u64 寄存器：%rd1, %rd2
.reg .pred %p<2>;    // 声明 2 个谓词寄存器：%p1, %p2

// 向量寄存器
.reg .v4.u32 %vr<1>; // 声明 1 个向量寄存器（4×u32）

// 特殊寄存器（只读）
mov.u32 %r1, %tid.x;       // 读取线程 X 索引
mov.u32 %r2, %ctaid.y;     // 读取块 Y 索引
mov.u64 %rd1, %clock64;    // 读取 64 位时钟
```

### 特殊寄存器完整列表

| 寄存器 | 说明 | 读写 |
|--------|------|------|
| `%tid.x/y/z` | 线程索引 | 只读 |
| `%ntid.x/y/z` | 块内线程数 | 只读 |
| `%ctaid.x/y/z` | 块索引 | 只读 |
| `%nctaid.x/y/z` | 网格内块数 | 只读 |
| `%smid` | SM 标识符 | 只读 |
| `%warpid` | Warp 标识符 | 只读 |
| `%laneid` | Lane 标识符 | 只读 |
| `%clock` | 32 位时钟 | 只读 |
| `%clock64` | 64 位时钟 | 只读 |
| `%pm0`, `%pm1` | 性能计数器 | 只读 |

## 1.7 第一个 PTX 程序

### 从 CUDA C 到 PTX

```cpp
// CUDA C 代码：向量加法
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### 编译为 PTX

```bash
# 编译 CUDA 代码为 PTX
nvcc -ptx vector_add.cu -o vector_add.ptx

# 查看 PTX 代码
cat vector_add.ptx
```

### 生成的 PTX 代码

```ptx
.version 7.0
.target sm_50, sm_52, sm_53, sm_60, sm_61, sm_62, sm_70, sm_72, sm_75, sm_80
.address_size 64

.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2,
    .param .u64 vector_add_param_3
) {
    .reg .b32 %r<3>;
    .reg .b64 %rd<7>;

    ld.param.u64 %rd1, [vector_add_param_0];
    ld.param.u64 %rd2, [vector_add_param_1];
    ld.param.u64 %rd3, [vector_add_param_2];
    ld.param.u64 %rd4, [vector_add_param_3];

    // 计算全局索引
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r2, %r1, %r2;
    mov.u32 %r1, %tid.x;
    add.u32 %r1, %r2, %r1;

    // 边界检查和计算
    cvt.u64.u32 %rd5, %r1;
    setp.ge.u32 %p1, %r1, %rd4;
    @%p1 bra done;

    // 加载、计算、存储
    shl.u64 %rd5, %rd5, 2;
    add.u64 %rd5, %rd1, %rd5;
    ld.global.f32 %f1, [%rd5];

    add.u64 %rd5, %rd2, %rd5;
    ld.global.f32 %f2, [%rd5];

    add.f32 %f1, %f1, %f2;

    add.u64 %rd5, %rd3, %rd5;
    st.global.f32 [%rd5], %f1;

done:
    ret;
}
```

## 1.8 小结

| 知识点 | 要点 |
|--------|------|
| PTX 定义 | 虚拟 ISA，类似 Java 字节码 |
| PTX vs SASS | PTX 与硬件无关，SASS 是真实机器码 |
| 学习价值 | 理解执行、性能优化、调试分析 |
| 程序结构 | 版本声明、函数定义、指令序列 |
| 内存空间 | global/shared/const/local/param |
| 寄存器 | 通用寄存器 + 特殊寄存器 |

**下一步**: 阅读 [02_PTX 数据类型.md](02_PTX 数据类型.md)
