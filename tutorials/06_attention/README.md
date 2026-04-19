# 第六课：Flash Attention 实现

## 1. Attention 机制

Attention 是现代深度学习（尤其是 Transformer）的核心运算：

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

其中:
- Q: Query 矩阵 (seq_len × d_k)
- K: Key 矩阵 (seq_len × d_k)
- V: Value 矩阵 (seq_len × d_v)
- d_k: 键维度
```

## 2. Attention 的内存挑战

**标准 Attention 问题：**
- 需要存储 N×N 的注意力矩阵
- 对于长序列（如 4K token），内存需求巨大
- 内存带宽成为瓶颈

## 3. Flash Attention 原理

Flash Attention 通过**分块计算**和**重计算**技术：
1. 将 Q, K, V 分块加载到共享内存
2. 在线计算 softmax（无需存储完整注意力矩阵）
3. 使用重计算技术减少内存访问

## 4. CUTE 实现 Flash Attention

使用 CUTE 可以：
- 优雅地处理分块 Layout
- 使用 Tensor Core 加速矩阵乘法
- 自动优化内存访问模式
