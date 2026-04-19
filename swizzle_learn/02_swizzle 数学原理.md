# 第二课：Swizzle 数学原理

## 2.1 二进制视角看 Swizzle

Swizzle 的核心公式非常简单：

```
swizzled_offset = offset ^ (offset >> shift)
```

其中 `^` 是 XOR（异或）运算，`>>` 是右移。

### XOR 运算复习

```
0 ^ 0 = 0
0 ^ 1 = 1
1 ^ 0 = 1
1 ^ 1 = 0
```

**关键性质**：XOR 是可逆的！
```
如果 a ^ b = c，那么 c ^ b = a
```

这意味着 Swizzle 是**可逆变换**，不会丢失信息。

## 2.2 完整的 Swizzle 公式

CUTE 中的 Swizzle 实际上更复杂一些：

```cpp
// 伪代码
swizzled_offset = offset ^ (((offset >> SShift) & mask) << BBits);
```

让我们分解这个公式：

### 步骤分解

假设 `Swizzle<2, 4, 3>`，即：
- `BBits = 2`（Bank 大小的对数）
- `NumBanks = 4`（Bank 数量的对数）
- `SShift = 3`（位移量）

对于输入 `offset = 13`（二进制：`001101`）：

```
步骤 1: 提取低位部分（用于确定 Bank 内偏移）
offset_low = offset & ((1 << BBits) - 1)
           = 13 & 3
           = 001101 & 000011
           = 000001 (十进制：1)

步骤 2: 右移提取高位部分
offset_high = offset >> SShift
            = 13 >> 3
            = 000001 (十进制：1)

步骤 3: 应用 mask（限制影响范围）
mask = (1 << NumBanks) - 1 = 15 (二进制：1111)
offset_masked = offset_high & mask
              = 1 & 15
              = 1

步骤 4: 左移到正确位置
xor_value = offset_masked << BBits
          = 1 << 2
          = 4 (二进制：000100)

步骤 5: XOR 得到最终结果
swizzled = offset ^ xor_value
         = 13 ^ 4
         = 001101 ^ 000100
         = 001001 (十进制：9)
```

## 2.3 为什么这个公式能避免 Bank Conflict？

关键在于：**XOR 只影响特定的位**。

### Bank 的确定

在 GPU 中，地址 `addr` 属于哪个 Bank 由下式决定：
```
bank_id = (addr / bytes_per_element) % num_banks
```

通常 `bytes_per_element = 4`（float），`num_banks = 32`。

简化为位运算：
```
bank_id = (addr >> 2) & 31  // 取第 2-6 位
```

### Swizzle 的魔法

Swizzle 的 XOR 操作会**翻转**某些位，从而改变 `bank_id`。

```
原始地址：    000100 (4) -> bank_id = (4>>2) & 31 = 1
XOR 值：      000100 (4)  // 由高位计算得到
Swizzled:     000000 (0) -> bank_id = (0>>2) & 31 = 0  // Bank 变了！
```

## 2.4 可逆性证明

为什么 Swizzle 是可逆的？

```cpp
// 正向：offset -> swizzled
swizzled = offset ^ (((offset >> SShift) & mask) << BBits);

// 反向：swizzled -> offset
// 关键：XOR 的低位部分在右移 SShift 后不变！
recovered = swizzled ^ (((swizzled >> SShift) & mask) << BBits);
// 因为 XOR 值 >> SShift == 0（当 SShift >= BBits + NumBanks）
// 所以 recovered = offset
```

**条件**：`SShift >= BBits + NumBanks` 时，XOR 不会影响高位。

## 2.5 地址变换可视化

### 8x8 矩阵的 Swizzle

```
原始布局 (行优先):
     Col0 Col1 Col2 Col3 Col4 Col5 Col6 Col7
Row0   0    1    2    3    4    5    6    7
Row1   8    9   10   11   12   13   14   15
Row2  16   17   18   19   20   21   22   23
Row3  24   25   26   27   28   29   30   31
Row4  32   33   34   35   36   37   38   39
Row5  40   41   42   43   44   45   46   47
Row6  48   49   50   51   52   53   54   55
Row7  56   57   58   59   60   61   62   63

Swizzle<2,3,3> 后的物理存储:
     Col0 Col1 Col2 Col3 Col4 Col5 Col6 Col7
Row0   0    1    2    3    8    9   10   11
Row1  16   17   18   19   24   25   26   27
Row2  32   33   34   35   40   41   42   43
Row3  48   49   50   51   56   57   58   59
Row4   4    5    6    7   12   13   14   15
Row5  20   21   22   23   28   29   30   31
Row6  36   37   38   39   44   45   46   47
Row7  52   53   54   55   60   61   62   63
```

注意：某些行被"交换"了，这就是 XOR 的效果！

## 2.6 数学本质总结

| 运算 | 作用 |
|------|------|
| `>> SShift` | 提取高位，决定是否翻转 |
| `& mask` | 限制翻转的范围 |
| `<< BBits` | 对齐到正确的位置 |
| `^` | 执行翻转（可逆！） |

**核心思想**：用地址的高位信息，XOR 翻转低位，从而改变 Bank 分配。

**下一步**：阅读 `03_参数详解.md`，理解每个参数的具体含义和如何选择。
