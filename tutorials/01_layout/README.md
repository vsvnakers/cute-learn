# 第一课：Layout 布局系统

## 1. 什么是 Layout？

Layout 是 CUTE 库最核心的概念。它是一个**从逻辑索引到线性偏移的映射函数**。

### 传统方式 vs CUTE Layout

**传统 CUDA 代码：**
```cpp
// 2D 矩阵访问 - 行优先
int idx = row * stride + col;
float val = data[idx];

// 3D 张量访问
int idx = n * stride1 + h * stride2 + w;
```

**CUTE Layout 方式：**
```cpp
#include <cute/layout.hpp>

// 定义一个 8x8 的 Layout
auto layout = make_layout(make_shape(8, 8));

// 访问元素 - Layout 自动计算偏移
auto tensor = make_tensor(ptr, layout);
auto val = tensor(2, 3);  // 自动计算 2*8+3=19
```

## 2. Layout 的数学定义

Layout 由三部分组成：
- **Shape**：每个维度的大小
- **Stride**：每个维度的步长
- **Size**：总元素数量

```
Layout: (shape, stride) -> offset
offset = sum(index[i] * stride[i])
```

## 3. 实战示例

让我通过完整代码展示 Layout 的使用：
