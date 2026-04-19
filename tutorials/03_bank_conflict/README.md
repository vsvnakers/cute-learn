# 第三课：Bank Conflict 解决方案

## 1. 什么是 Bank Conflict？

### GPU 共享内存架构

GPU 的共享内存（Shared Memory）被划分为多个等宽的内存 Bank：
- **Bank 数量**：通常 32 个（Ampere/Hopper 架构）
- **Bank 宽度**：4 字节（32-bit）
- **总带宽**：32 Bank × 4 字节 = 128 字节/周期

### Bank Conflict 发生条件

当**同一个 warp 中的多个线程**访问**同一个 Bank**的不同地址时，会发生 Bank Conflict。

```
Warp 中的 32 个线程：
线程 0: 访问地址 0   (Bank 0)
线程 1: 访问地址 4   (Bank 1)
线程 2: 访问地址 8   (Bank 2)
...
线程 31: 访问地址 124 (Bank 31)
结果：无 Bank Conflict，所有访问并行完成

线程 0: 访问地址 0   (Bank 0)
线程 1: 访问地址 32  (Bank 0) ← 冲突！
线程 2: 访问地址 64  (Bank 0) ← 冲突！
线程 3: 访问地址 96  (Bank 0) ← 冲突！
结果：4-way Bank Conflict，需要 4 个周期串行执行
```

## 2. Bank Conflict 的类型

- **无冲突（No Conflict）**：每个 Bank 最多被访问一次
- **2-way Conflict**：一个 Bank 被 2 个线程访问
- **4-way Conflict**：一个 Bank 被 4 个线程访问
- **32-way Conflict**：最坏情况，所有线程访问同一 Bank

## 3. 解决 Bank Conflict 的方法

### 方法 1：Padding（填充）
在每行末尾添加额外元素，改变后续行的起始 Bank。

### 方法 2：Swizzle（置换）
使用 XOR 运算重新映射地址，均匀分散访问。

### 方法 3：Layout 重排
改变数据的存储布局，如使用 Tile Layout。
