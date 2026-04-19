# 第二课：Swizzle 技术

## 1. 什么是 Swizzle？

Swizzle 是一种**内存地址变换技术**，用于优化 GPU 共享内存访问模式，避免 Bank Conflict。

### 为什么需要 Swizzle？

GPU 的共享内存被分成多个 Bank（通常 32 个），当多个线程同时访问同一个 Bank 的不同地址时，会发生 Bank Conflict，导致串行化访问。

**没有 Swizzle 的问题：**
```
线程 0,1,2,3 访问地址 0,1,2,3 -> 可能在同一个 Bank！
结果：4 次串行访问
```

**使用 Swizzle 后：**
```
线程 0,1,2,3 访问地址 0,8,4,12 -> 分散到不同 Bank
结果：1 次并行访问
```

## 2. Swizzle 的数学原理

Swizzle 通过对地址的低位进行 XOR 运算来重新排列地址：

```
swizzled_offset = offset ^ ((offset >> shift) & mask)
```

## 3. CUTE 中的 Swizzle

CUTE 提供了多种 Swizzle 实现：
- `swizzle<2,4,2>`：参数化 Swizzle
- `XOR_swizzle`：XOR 基础 Swizzle
- TMA Swizzle：用于 Hopper 架构的 TMA 单元
