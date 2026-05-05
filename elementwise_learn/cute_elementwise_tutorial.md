# CuTe Elementwise 算子原理详解

## 1. 从朴素 kernel 到 CuTe kernel 的演进

### 1.1 朴素写法的问题

```cuda
// 朴素 kernel：每个线程处理 1 个 float（4 字节）
__global__ void naive_add(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}
```

问题：每个线程只发 1 条 `LDG.32`（32-bit load），而 GPU 的内存总线宽度是 128-bit。
这就像用吸管喝可乐——一次只喝一滴，浪费了整根吸管的容量。

### 1.2 CuTe 的思路

CuTe 的核心思想：**让每个线程一次加载 128-bit（4 个 float），用满内存总线。**

```
朴素:   线程0→[float] 线程1→[float] 线程2→[float] 线程3→[float]  (4条32-bit指令)
CuTe:   线程0→[float4]                                               (1条128-bit指令)
```

---

## 2. CuTe 核心概念图解

### 2.1 Tensor（张量）

CuTe 的 Tensor = **数据指针 + Layout（布局描述）**

```
Tensor = Pointer + Layout

Pointer: 指向数据的起始地址（GMEM / SMEM / Register）
Layout:  描述"逻辑坐标"到"物理地址"的映射关系

举例：一个有 8 个 float 的 1D 张量
  Pointer → [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  Layout  → 坐标 0→地址0, 坐标 1→地址1, ..., 坐标 7→地址7
```

关键：Layout 不持有数据，只描述"怎么访问数据"。同一个指针可以有多种 Layout。

### 2.2 make_tensor — 创建张量

```cpp
// 创建一个指向 GPU 全局内存的 1D 张量，有 N 个 float
Tensor tensor_A = make_tensor(
    make_gmem_ptr(d_A),    // 数据指针（GPU 全局内存）
    make_layout(N)         // 布局：N 个元素的 1D 数组
);

// 此时 tensor_A 的形状是 (N,)
// tensor_A(i) 等价于 d_A[i]
```

### 2.3 tiled_divide — 分块

这是 CuTe 最核心的操作之一。它把一个 1D 张量"折叠"成 2D：

```
原始张量 (N=8):
  [a0, a1, a2, a3, a4, a5, a6, a7]
   ←───── 连续内存 ─────────────→

tiled_divide(TILE=4) 后变成 (4, 2):
  ┌─────────────┬─────────────┐
  │ a0  a1  a2  a3 │ a4  a5  a6  a7 │
  │  tile 0        │  tile 1        │
  └─────────────┴─────────────┘
   第0维(TILE=4)     第1维(num_tiles=2)

视觉化：
  坐标 (i, j) → 原始地址 i + j*4
  tiled_A(2, 1) = a6   // tile 1 的第 2 个元素
```

**为什么要分块？** 因为 CUDA 的一个 thread block 处理一个 tile。
分块后，`blockIdx.x` 直接对应第几个 tile，非常自然。

代码对应：
```cpp
// 1D 张量 (N,) → 2D 张量 (1024, num_tiles)
Tensor tiled_A = tiled_divide(
    make_tensor(make_gmem_ptr(d_A), make_layout(N)),  // 原始 1D 张量
    make_shape(Int<1024>{})                            // tile 大小
);
// 现在 tiled_A 的形状是 (1024, num_tiles)
// tiled_A(i, j) = d_A[j*1024 + i]
```

### 2.4 Layout — 坐标到地址的映射

Layout 是 CuTe 的灵魂。它描述"逻辑索引"到"物理偏移"的映射。

```
Layout = (Shape, Stride)

Shape:  每个维度有多少个元素
Stride: 每个维度移动 1，物理地址变化多少

例子：(4, 2) 的 Layout，stride 为 (1, 4)
  坐标 (0,0) → 0*1 + 0*4 = 0
  坐标 (1,0) → 1*1 + 0*4 = 1
  坐标 (3,0) → 3*1 + 0*4 = 3
  坐标 (0,1) → 0*1 + 1*4 = 4
  坐标 (1,1) → 1*1 + 1*4 = 5
  坐标 (3,1) → 3*1 + 1*4 = 7

这正好是列优先（column-major）的 4×2 矩阵！
```

CuTe 中的 `make_layout` 会自动选择 stride：
```cpp
make_layout(Int<4>{})           // shape=(4,), stride=(1,) — 默认连续
make_layout(Int<4>{}, Int<8>{}) // shape=(4,), stride=(8,) — 跳跃访问
```

### 2.5 Copy_Atom — 拷贝原子操作

Copy_Atom 定义了"一次内存操作怎么做"：

```
Copy_Atom = 操作类型 + 数据类型

操作类型决定：
  - 一次加载多少字节
  - 用什么指令（LDG.32 / LDG.64 / LDG.128）
  - 是否需要对齐

我们的选择：
  UniversalCopy<uint4>  → 一次加载 128 bit（16 字节）
  数据类型 float        → 每个 float 4 字节
  所以一次加载 4 个 float
```

```cpp
// Copy_Atom<操作类型, 数据类型>
using Atom = Copy_Atom<UniversalCopy<uint4>, float>;
//                         ↑                    ↑
//                    用 128-bit 加载         操作 float 类型数据
//                    (uint4 = 4×32bit)
```

**UniversalCopy vs AutoVectorizingCopy：**
- `UniversalCopy<uint4>`: 强制使用 128-bit 指令，你告诉它用多宽
- `AutoVectorizingCopy`: 自动选择最宽的指令（可能选错）

### 2.6 make_tiled_copy — 构建分块拷贝

`make_tiled_copy` 把一个 Copy_Atom "铺满" 到整个 tile 上：

```
输入：
  Copy_Atom: 一次处理 4 个 float（128-bit）
  ThreadLayout: 256 个线程，排成 1D
  ValueLayout: 每个线程处理 4 个值

输出：
  TiledCopy: 一个描述"256 个线程如何协作搬运 1024 个 float"的计划

视觉化（1024 个 float 的 tile）：
  线程 0:   [f0  f1  f2  f3  ]     ← 一个 float4
  线程 1:   [f4  f5  f6  f7  ]
  线程 2:   [f8  f9  f10 f11 ]
  ...
  线程 255: [f1020 f1021 f1022 f1023]

  每个线程发 1 条 LDG.128，256 条指令搬运 1024 个 float = 4KB
```

```cpp
// 定义线程布局：256 个线程排成 1D
Layout thr_layout = make_layout(Int<256>{});   // (256,) → 线程ID

// 定义值布局：每个线程处理 4 个连续值
Layout val_layout = make_layout(Int<4>{});     // (4,)   → 值索引

// 组合成 TiledCopy
TiledCopy tiled_copy = make_tiled_copy(
    Atom{},        // 拷贝原子：128-bit float4
    thr_layout,    // 线程布局：256 threads
    val_layout     // 值布局：4 values/thread
);
```

### 2.7 get_thread_slice — 获取线程的切片

```
整个 tile (1024 个元素) 被 256 个线程瓜分：
  ┌──────────────────────────────────────────┐
  │  线程0: [0..3]  线程1: [4..7]  ...  线程255: [1020..1023]  │
  └──────────────────────────────────────────┘

get_thread_slice(threadIdx.x) 返回当前线程负责的那部分"视图"。
它不拷贝数据，只描述"这个线程应该访问哪些元素"。
```

```cpp
auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

// thr_copy.partition_S(tile_A) 把 tile_A 切成当前线程负责的部分
Tensor thr_A = thr_copy.partition_S(tile_A);
// thr_A 的形状大约是 (4,) — 当前线程的 4 个 float
```

### 2.8 make_fragment_like — 创建寄存器张量

```
GMEM（全局内存）→ Registers（寄存器）→ GMEM
     读取            计算             写回

fragment 是驻留在寄存器中的张量，形状和 thr_A 一致。
```

```cpp
Tensor frag_A = make_fragment_like(thr_A);  // 形状 (4,)，存在寄存器里
Tensor frag_B = make_fragment_like(thr_B);
Tensor frag_C = make_fragment_like(thr_C);
```

### 2.9 copy — 执行拷贝

```cpp
// 从 GMEM 加载到寄存器（LDG.128）
copy(tiled_copy, thr_A, frag_A);   // A[i..i+3] → 寄存器
copy(tiled_copy, thr_B, frag_B);   // B[i..i+3] → 寄存器

// 计算（纯寄存器操作，极快）
for (int i = 0; i < 4; ++i)
    frag_C(i) = frag_A(i) + frag_B(i);

// 从寄存器写回 GMEM（STG.128）
copy(tiled_copy, frag_C, thr_C);   // 寄存器 → C[i..i+3]
```

---

## 3. 完整数据流图

```
Host 端准备：
  ┌─────────────────────────────────────────────────┐
  │  d_A ──→ make_gmem_ptr ──→ make_tensor ──→ tiled_divide  │
  │         (原始指针)      (1D张量)        (2D张量)       │
  │         (N,)          (1024, num_tiles)             │
  └─────────────────────────────────────────────────┘

Kernel 端执行（每个 block）：
  ┌─────────────────────────────────────────────────┐
  │  tiled_A(_, blockIdx.x)  ──→  取出第 blockIdx.x 个 tile     │
  │         (1024,)                                      │
  │              │                                       │
  │              ▼                                       │
  │  thr_copy.partition_S  ──→  取出当前线程的 4 个元素      │
  │         (4,)                                         │
  │              │                                       │
  │              ▼                                       │
  │  copy(tiled_copy, thr, frag) ──→  LDG.128 加载到寄存器  │
  │         (4,) in registers                            │
  │              │                                       │
  │              ▼                                       │
  │  frag_C = frag_A + frag_B ──→  寄存器加法              │
  │              │                                       │
  │              ▼                                       │
  │  copy(tiled_copy, frag, thr) ──→  STG.128 写回 GMEM   │
  └─────────────────────────────────────────────────┘
```

---

## 4. 为什么这样写能跑满带宽？

### 4.1 向量化访问

```
朴素 float:   每线程 4字节  × 256线程 = 1KB / 32条指令
CuTe float4:  每线程 16字节 × 256线程 = 4KB / 1条指令  ← 32x 更少的指令数
```

GPU 内存控制器以 128-byte cache line 为单位工作。
一条 LDG.128 发出 16 字节请求，128/16 = 8 条指令就能填满一个 cache line。
而 LDG.32 需要 32 条指令。指令数少了，调度开销也少了。

### 4.2 消除 grid-stride loop

```
Grid-stride loop:
  gridDim = 112 blocks
  每个 block 循环 256K/112 ≈ 2286 次
  每次循环：判断条件 + 计算偏移 + 创建张量 + 执行拷贝

CuTe 无循环:
  gridDim = 256K blocks
  每个 block 只执行 1 次：取 tile → 分区 → 拷贝 → 计算 → 拷贝
  零循环开销
```

### 4.3 对齐的内存访问

`cudaMalloc` 返回 256 字节对齐的指针。float4 只需要 16 字节对齐。
tiled_divide 保证每个 tile 的起始地址是 `tile_idx * 1024 * 4` 字节，1024*4=4096，远超 16 字节对齐要求。
所以每个 LDG.128 都是对齐的，不会触发额外的 cache line 拆分。

### 4.4 合并访问（Coalesced Access）

GPU 的 32 个线程组成一个 warp。同一个 warp 的内存请求会被合并。

```
Warp 内 32 个线程的 LDG.128：
  线程 0:  加载地址 [0x0000, 0x000F]  (16字节)
  线程 1:  加载地址 [0x0010, 0x001F]
  ...
  线程 31: 加载地址 [0x01F0, 0x01FF]

合并后：一次请求 32 × 16 = 512 字节 = 4 个 cache line
这是 GPU 内存系统最高效的工作模式。
```

---

## 5. 关键数值分析

### RTX 3060 Laptop 参数
- SM 数量: 28
- 峰值内存带宽: ~336 GB/s (GDDR6, 192-bit bus)
- L2 Cache: 3 MB

### Elementwise Add 的访存量
- 读 A: N × 4 bytes
- 读 B: N × 4 bytes
- 写 C: N × 4 bytes
- 总计: N × 12 bytes

### N=256M (1GB/tensor) 时
- 总访存: 256M × 12 = 3072 MB = 3 GB
- 理论最短时间: 3072 MB / 336 GB/s = 9.14 ms
- 实测时间: 10.8 ms
- 实际带宽: 3072 MB / 10.8 ms = 284 GB/s → **88.7% of peak**

### 为什么达不到 100%？
1. **TLB miss**: 大数组访问可能触发页表查找
2. **L2 cache 污染**: A/B/C 三个数组在 L2 中互相驱逐
3. **指令开销**: 即使无循环，kernel 启动/退出、tile 索引计算也有开销
4. **Warp 调度**: 不是所有 warp 都能同时发出内存请求

---

## 6. CuTe vs 朴素写法对比

| 特性 | 朴素 float | CuTe float4 |
|------|-----------|-------------|
| 每线程加载宽度 | 32-bit (4B) | 128-bit (16B) |
| 指令数 (1024元素) | 1024 条 LDG.32 | 256 条 LDG.128 |
| 边界处理 | if (idx < N) | CuTe predicated copy |
| 分块逻辑 | 手动计算 | tiled_divide 自动 |
| 线程-数据映射 | 手动 index | TiledCopy 自动 |
| 带宽利用率 | ~50-60% | ~88% |

---

## 7. 代码执行流程总结

```
1. Host: make_tensor(d_A, N)          → 创建 1D 张量
2. Host: tiled_divide(tensor, 1024)   → 折叠为 (1024, num_tiles) 2D 张量
3. Host: make_tiled_copy(atom, 256, 4) → 定义"256线程×4值"的拷贝计划
4. Host: kernel<<<num_tiles, 256>>>(tiled_copy, tiled_A, tiled_B, tiled_C)
5. Kernel: 每个 block 取 tile_idx = blockIdx.x
6. Kernel: tiled_A(_, tile_idx)       → 取出第 idx 个 tile
7. Kernel: partition_S → 每线程 4 个元素
8. Kernel: copy(GMEM→Reg)             → LDG.128 加载
9. Kernel: frag_C = frag_A + frag_B   → 寄存器加法
10. Kernel: copy(Reg→GMEM)            → STG.128 写回
```

---

## 8. 扩展阅读

- CuTe 源码: `cutlass/include/cute/`
- 核心文件:
  - `tensor.hpp` — Tensor 定义
  - `layout.hpp` — Layout 定义（Shape + Stride）
  - `copy.hpp` — copy 算法
  - `algorithm/copy.hpp` — copy_if, copy 等
  - `atom/copy_atom.hpp` — Copy_Atom 定义
- 示例: `cutlass/examples/cute/tutorial/tiled_copy.cu`
