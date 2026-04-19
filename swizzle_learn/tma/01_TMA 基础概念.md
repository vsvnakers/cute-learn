# 第一课：TMA 基础概念

## 1.1 什么是 TMA？

**TMA (Tensor Memory Accelerator)** 是 NVIDIA Hopper 架构引入的专用硬件单元。

### TMA 的核心功能

```
┌─────────────────────────────────────────────────────────────┐
│                    TMA 硬件单元                              │
├─────────────────────────────────────────────────────────────┤
│  1. 地址计算引擎    →  硬件自动计算复杂的多维地址            │
│  2. 边界检查单元    →  硬件处理边界，无需软件分支            │
│  3. Swizzle 单元    →  硬件内置 Bank Conflict 避免           │
│  4. 异步拷贝引擎    →  独立的 DMA 通道，不占用 SM 资源        │
│  5. mbarrier 支持   →  硬件同步机制                          │
└─────────────────────────────────────────────────────────────┘
```

## 1.2 为什么需要 TMA？

### 传统方式的问题

在 Ampere 及更早的架构中，共享内存拷贝需要：

```cuda
// 传统方式：软件管理一切
__global__ void traditional_copy(float* gmem, float* smem_out, int N) {
    __shared__ float shared_mem[256];

    int tid = threadIdx.x;

    // 1. 软件计算地址
    int global_idx = blockIdx.x * 256 + tid;

    // 2. 软件边界检查
    if (global_idx < N) {
        // 3. 软件处理 Swizzle（XOR 运算）
        int swizzled_idx = swizzle_function(tid);

        // 4. 同步加载
        shared_mem[swizzled_idx] = gmem[global_idx];
    }

    __syncthreads();  // 需要显式同步

    // 5. 计算输出地址
    int out_idx = blockIdx.x * 256 + tid;
    if (out_idx < N) {
        // 6. 再次 Swizzle
        int out_swizzled = swizzle_function(tid);
        smem_out[out_idx] = shared_mem[out_swizzled];
    }
}
```

**问题**：
- 占用 SM 指令带宽
- 需要显式同步
- 地址计算消耗寄存器
- 边界检查有分支开销

### TMA 方式

```cuda
// TMA 方式：硬件管理
__global__ void tma_copy(TensorMap desc, float* smem_out, int N) {
    __shared__ float shared_mem[256];
    __shared__ uint64_t mbarrier;

    // 初始化 mbarrier
    if (threadIdx.x == 0) {
        mbarrier_init(&mbarrier, 256);  // 等待 256 字节
    }
    __syncthreads();

    // TMA 拷贝：硬件处理一切！
    // - 地址计算（硬件）
    // - 边界检查（硬件）
    // - Swizzle（硬件）
    // - 异步拷贝（硬件 DMA）
    tma_load(desc, &mbarrier, shared_mem, blockIdx.x * 256);

    // 等待 TMA 完成
    mbarrier_wait(&mbarrier);

    // 现在 shared_mem 已就绪，可以计算
}
```

**优势**：
- 不占用 SM 指令带宽
- 硬件异步拷贝
- 自动边界检查
- 内置 Swizzle

## 1.3 TMA 描述符 (TensorMap)

TMA 操作需要一个**描述符**，包含所有拷贝信息。

### 描述符包含的信息

```cpp
struct TmaDescriptor {
    // 全局内存信息
    void* global_address;           // 全局内存基地址
    uint64_t gmem_strides[5];       // 各维度 stride

    // 张量形状
    uint64_t tensor_shape[5];       // 完整张量形状

    // Box（拷贝块）形状
    uint64_t box_shape[5];          // 每次拷贝的大小

    // 元素类型
    uint32_t element_type;          // e.g., F32, F16, S8

    // Swizzle 配置
    uint32_t swizzle_mode;          // B32, B64, B128
    uint32_t elem_bytes;            // 每个元素字节数

    // 其他配置
    uint32_t interleave_layout;     // interleaved 布局
    uint32_t padding;               // padding 配置
};
```

### 创建描述符

CUTE 提供了高级 API 来创建描述符：

```cpp
// CUTE 方式：使用 make_tma_copy
auto gmem_tensor = make_tensor(make_gmem_ptr(data),
                                make_shape(N, M),
                                make_stride(M, 1));

// 创建 TMA 拷贝算子
auto tma_copy = make_tma_copy(
    gmem_tensor,              // 全局内存张量
    smem_layout,              // 共享内存布局
    Swizzle<2, 5, 3>{},       // Swizzle 配置
    1                         // multicast 数量
);

// 使用
copy(tma_copy, gmem_coord, smem_tensor);
```

## 1.4 mbarrier 机制

TMA 使用**硬件 mbarrier**进行同步。

### mbarrier 工作原理

```
线程发出 TMA 请求
       ↓
硬件开始异步拷贝
       ↓
线程继续其他工作（或等待）
       ↓
TMA 完成，硬件更新 mbarrier
       ↓
等待的线程被唤醒
```

### 使用示例

```cuda
__shared__ uint64_t mbarrier;

// 初始化：指定要等待的事务数量
if (threadIdx.x == 0) {
    // 参数：mbarrier 指针，要等待的事务数
    mbarrier_init(&mbarrier, transaction_count);
}
__syncthreads();

// 发出 TMA 请求
tma_load(desc, &mbarrier, smem_ptr, coord);

// 等待完成
// 方式 1: 阻塞等待
mbarrier_wait(&mbarrier);

// 方式 2: 带超时的等待
while (!mbarrier_try_wait(&mbarrier, timeout)) {
    // 可以做其他事情
}
```

## 1.5 TMA 支持的操作

### 拷贝方向

| 操作 | 方向 | 描述 |
|------|------|------|
| TMA_LOAD | Global → Shared | 从全局内存加载到共享内存 |
| TMA_STORE | Shared → Global | 从共享内存存储到全局内存 |
| TMA_LOAD_MULTICAST | Global → Shared (多副本) | 加载到多个 CTA 的共享内存 |

### 维度支持

| 维度 | 操作 | 坐标参数 |
|------|------|----------|
| 1D | `tma_load_1d` | `(coord0)` |
| 2D | `tma_load_2d` | `(coord0, coord1)` |
| 3D | `tma_load_3d` | `(coord0, coord1, coord2)` |
| 4D | `tma_load_4d` | `(coord0, coord1, coord2, coord3)` |
| 5D | `tma_load_5d` | `(coord0, coord1, coord2, coord3, coord4)` |

## 1.6 TMA 内置 Swizzle

TMA 的 Swizzle 是**硬件自动应用**的。

### 配置方式

```cpp
// CUTE 中，Swizzle 配置是 TMA 描述符的一部分
auto tma_copy = make_tma_copy(
    gmem_tensor,
    smem_layout,
    Swizzle<2, 5, 3>{},    // ← 这里配置 Swizzle
    1
);

// 硬件会根据 Swizzle 参数自动：
// 1. 计算 swizzled 共享内存地址
// 2. 配置 TMA 描述符中的 swizzle_mode 字段
// 3. 在拷贝时自动应用
```

### 与软件 Swizzle 的对比

```
软件 Swizzle (Ampere):
  线程计算 XOR → 写入 swizzled 地址
  └── 消耗指令
  └── 消耗寄存器

TMA Swizzle (Hopper):
  线程发出请求 → 硬件计算 XOR → 写入
  └── 不消耗 SM 指令
  └── 零开销
```

## 1.7 完整的 TMA 拷贝流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    TMA 拷贝完整流程                              │
└─────────────────────────────────────────────────────────────────┘

1. 主机端创建 TensorMap 描述符
   ├─ 设置全局内存地址
   ├─ 设置张量形状和 stride
   ├─ 设置 Box 形状（拷贝块大小）
   ├─ 配置 Swizzle 参数 ← 关键！
   └─ 拷贝到设备

2. 设备端初始化 mbarrier
   └─ mbarrier_init(&mbar, expect_count)

3. 发出 TMA 请求
   ├─ tma_load(desc, &mbar, smem_ptr, coord)
   ├─ 硬件计算全局地址
   ├─ 硬件边界检查
   └─ 硬件异步拷贝（含 Swizzle）

4. 等待完成
   ├─ mbarrier_wait(&mbar)
   └─ 数据已就绪

5. 使用数据
   └─ shared_mem 中的数据可直接使用
```

## 1.8 小结

| 概念 | 要点 |
|------|------|
| TMA | 硬件加速的内存拷贝单元 |
| TensorMap | TMA 描述符，包含所有拷贝信息 |
| mbarrier | 硬件同步机制 |
| TMA Swizzle | 硬件内置，零开销 |
| 优势 | 不占用 SM 资源，自动边界检查 |

**下一步**: 阅读 [02_TMA_Swizzle 参数详解.md](02_TMA_Swizzle 参数详解.md)
