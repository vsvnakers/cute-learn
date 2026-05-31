/**
 * ============================================================================
 * CuTe + MMA 教程 01: Layout 基础
 * ============================================================================
 *
 * 本文件讲解 CuTe 中最核心的概念：Layout（布局）
 * Layout = Shape（形状） + Stride（步长）
 *
 * CuTe 的 Layout 是理解 MMA、Copy 等一切操作的基础。
 * 一个 Layout 描述了：给定 N 维逻辑坐标，如何映射到线性内存偏移。
 *
 * 公式：offset = sum(coord_i * stride_i)
 *
 * 编译：make 01_layout_basics
 * 运行：./01_layout_basics
 */

#include <iostream>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// ============================================================================
// 辅助函数：打印 Layout 信息
// ============================================================================

template <class Layout>
void print_layout_info(const char* name, Layout layout) {
    std::cout << "  " << name << ":" << std::endl;
    std::cout << "    Layout = " << layout << std::endl;
    std::cout << "    Shape  = " << shape(layout) << std::endl;
    std::cout << "    Stride = " << stride(layout) << std::endl;
    std::cout << "    Rank   = " << rank(layout) << std::endl;       // 维度数量
    std::cout << "    Size   = " << size(layout) << std::endl;       // 总元素数
    std::cout << "    Depth  = " << depth(layout) << std::endl;      // 嵌套深度
    std::cout << std::endl;
}

// ============================================================================
// 1. make_layout 基础
// ============================================================================
// make_layout 是创建 Layout 的核心 API
// 它的默认行为是：当只给 Shape 不给 Stride 时，使用 Row-Major（行优先）步长
//
// make_layout(shape) 的默认 stride 生成规则：
//   对于 shape (M, N, K, ...):
//   stride = (N*K*..., K*..., ..., 1)  -- Row-Major
//
// 注意：CuTe 中的 Layout 使用编译期整数常量（如 _4, _8 等），
//       这些是 cute::integral_constant 类型，可以在编译期求值

void test_basic_layout() {
    std::cout << "=== 1. make_layout 基础 ===" << std::endl;
    std::cout << std::endl;

    // ---- 1D Layout ----
    // make_layout(make_shape(N)) 默认生成 stride=1 的 1D layout
    // 等价于 make_layout(make_shape(N), make_stride(1))
    auto layout_1d = make_layout(make_shape(Int<8>{}));
    print_layout_info("1D Layout (8)", layout_1d);
    // 输出：Layout = (8):(1)
    // 含义：8 个元素，stride=1，连续内存

    // ---- 2D Layout - Row-Major ----
    // make_layout(make_shape(M, N)) 默认使用 Row-Major stride
    // Row-Major: stride = (N, 1)，即行内连续
    auto layout_2d_row = make_layout(make_shape(Int<4>{}, Int<8>{}));
    print_layout_info("2D Row-Major (4x8)", layout_2d_row);
    // 输出：Layout = (4,8):(8,1)
    // 含义：4 行 8 列，行内 stride=1，跨行 stride=8

    // ---- 2D Layout - Column-Major ----
    // 手动指定 Column-Major stride
    // Column-Major: stride = (1, M)，即列内连续
    auto layout_2d_col = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                     make_stride(Int<1>{}, Int<4>{}));
    print_layout_info("2D Column-Major (4x8)", layout_2d_col);
    // 输出：Layout = (4,8):(1,4)
    // 含义：4 行 8 列，列内 stride=1，跨列 stride=4

    // ---- 3D Layout ----
    // 用于 MMA 中的 (M, N, K) 形状
    // 默认 Row-Major: stride = (N*K, K, 1)
    auto layout_3d = make_layout(make_shape(Int<16>{}, Int<8>{}, Int<8>{}));
    print_layout_info("3D Layout (16x8x8)", layout_3d);
    // 输出：Layout = (16,8,8):(64,8,1)
    // 含义：(M=16, N=8, K=8)，M 维 stride=64，K 维 stride=1

    // ---- 运行时 Layout ----
    // 使用 int 而非 Int<> 创建运行时 Layout
    auto layout_runtime = make_layout(make_shape(4, 8));
    print_layout_info("Runtime Layout (4x8)", layout_runtime);
}

// ============================================================================
// 2. Shape 和 Stride 的独立创建
// ============================================================================
// make_shape: 创建形状，描述每个维度的大小
// make_stride: 创建步长，描述每个维度的内存跳跃

void test_shape_stride() {
    std::cout << "=== 2. Shape 和 Stride ===" << std::endl;
    std::cout << std::endl;

    // ---- Shape ----
    auto s1 = make_shape(Int<4>{}, Int<8>{});    // 编译期常量
    auto s2 = make_shape(4, 8);                   // 运行时值
    std::cout << "  Shape (编译期): " << s1 << std::endl;
    std::cout << "  Shape (运行时): " << s2 << std::endl;

    // ---- Stride ----
    auto st1 = make_stride(Int<8>{}, Int<1>{});   // Row-Major stride
    auto st2 = make_stride(Int<1>{}, Int<4>{});   // Column-Major stride
    std::cout << "  Stride (Row-Major): " << st1 << std::endl;
    std::cout << "  Stride (Col-Major): " << st2 << std::endl;

    // ---- 坐标到偏移的映射 ----
    // 使用 cute::coshape 和手动计算来演示
    auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                              make_stride(Int<8>{}, Int<1>{}));
    std::cout << std::endl;
    std::cout << "  Layout: " << layout << std::endl;
    std::cout << "  坐标 -> 偏移 映射:" << std::endl;
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 8; n++) {
            auto offset = layout(m, n);  // 计算 (m,n) 的线性偏移
            std::cout << "    (" << m << "," << n << ") -> " << offset << std::endl;
        }
    }
    std::cout << std::endl;
}

// ============================================================================
// 3. Layout 的嵌套结构 (Hierarchical Layout)
// ============================================================================
// CuTe 支持嵌套 Shape/Stride，用于描述线程和值的映射关系
// 这在 MMA 和 Copy 中非常重要
//
// 例如 SM80 MMA 中 A 矩阵的 Layout:
//   SM80_16x8_Row = Layout<Shape <Shape < _4,_8>, _2>,
//                          Stride<Stride<_32,_1>,_8>>
//   含义：
//     - 外层 shape (4,8) x 2：表示 32 个线程，每个线程 2 个值
//     - stride (32,1) x 8：线程维度 stride=32,1，值维度 stride=8

void test_nested_layout() {
    std::cout << "=== 3. 嵌套 Layout (Hierarchical) ===" << std::endl;
    std::cout << std::endl;

    // ---- 简单嵌套 ----
    // (2, (3,4)) : ((5, 1), (20, 5))
    // 外层 shape = (2, 12)，内层拆分 12 -> (3,4)
    auto nested = make_layout(
        make_shape(Int<2>{}, make_shape(Int<3>{}, Int<4>{})),
        make_stride(make_stride(Int<5>{}, Int<1>{}), make_stride(Int<20>{}, Int<5>{}))
    );
    print_layout_info("嵌套 Layout", nested);

    // ---- SM80 MMA A 矩阵 Layout 示例 ----
    // 这是 SM80_16x8_Row 的定义：
    // (T32,V2) -> (M16,N8)
    // T32 = (4,8) 表示 32 个线程的 2D 排列
    // V2  = (2,2) 表示每个线程持有 4 个值
    using SM80_16x8_Row = Layout<Shape <Shape <_4, _8>, Shape <_2, _2>>,
                                 Stride<Stride<_32, _1>, Stride<_16, _8>>>;

    std::cout << "  SM80_16x8_Row Layout (MMA A 矩阵布局):" << std::endl;
    std::cout << "    Layout = " << SM80_16x8_Row{} << std::endl;
    std::cout << "    Shape  = " << shape(SM80_16x8_Row{}) << std::endl;
    std::cout << "    Size   = " << size(SM80_16x8_Row{}) << std::endl;
    std::cout << std::endl;

    // 解读 SM80_16x8_Row:
    // - 总共 32 个线程 (4*8)，每个线程 4 个值 (2*2)
    // - 线程维度: (4,8):(32,1) -> 线程编号 = (tid/8)*32 + (tid%8)*1
    // - 值维度:   (2,2):(16,8) -> 值在矩阵中的偏移
    //
    // 对于线程 tid 和值 val:
    //   线程 2D 坐标: (tid/8, tid%8)
    //   值 2D 坐标:   (val/2*16 + val%2*8)
    //   最终 M 坐标:   (tid/8)*4 + (val/2)*16
    //   最终 N 坐标:   (tid%8)*1 + (val%2)*8

    // 手动验证几个坐标
    auto layout_A = SM80_16x8_Row{};
    std::cout << "  SM80_16x8_Row 坐标映射示例:" << std::endl;
    // 线程 0 的 4 个值
    for (int thr = 0; thr < 2; thr++) {
        for (int val = 0; val < 4; val++) {
            auto offset = layout_A(thr, val);
            std::cout << "    (thr=" << thr << ", val=" << val << ") -> offset=" << offset << std::endl;
        }
    }
    std::cout << std::endl;
}

// ============================================================================
// 4. Layout 的常用操作
// ============================================================================

void test_layout_operations() {
    std::cout << "=== 4. Layout 常用操作 ===" << std::endl;
    std::cout << std::endl;

    auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                              make_stride(Int<8>{}, Int<1>{}));

    // size: 总元素数
    std::cout << "  size(layout)   = " << size(layout) << std::endl;

    // shape: 获取形状
    std::cout << "  shape(layout)  = " << shape(layout) << std::endl;

    // stride: 获取步长
    std::cout << "  stride(layout) = " << stride(layout) << std::endl;

    // rank: 维度数
    std::cout << "  rank(layout)   = " << rank(layout) << std::endl;

    // depth: 嵌套深度
    std::cout << "  depth(layout)  = " << depth(layout) << std::endl;

    // 访问特定维度
    // size<0>: 第 0 维大小
    std::cout << "  size<0>(layout) = " << size<0>(layout) << std::endl;
    std::cout << "  size<1>(layout) = " << size<1>(layout) << std::endl;

    // stride<0>: 第 0 维步长
    std::cout << "  stride<0>(layout) = " << stride<0>(layout) << std::endl;
    std::cout << "  stride<1>(layout) = " << stride<1>(layout) << std::endl;

    // ---- Layout composition (组合) ----
    // compose 将两个 layout 组合：layout_b(layout_a(coord))
    auto inner = make_layout(make_shape(Int<2>{}, Int<4>{}));
    auto outer = make_layout(make_shape(Int<2>{}, Int<2>{}));
    auto composed = composition(layout, inner);
    std::cout << std::endl;
    std::cout << "  composition 结果: " << composed << std::endl;

    // ---- Layout complement (补集) ----
    // complement: 给定一个 layout，求其在更大空间中的"补"
    // 用于将一个小 layout 扩展到更大的空间
    auto comp = complement(layout, Int<64>{});
    std::cout << "  complement(layout, 64) = " << comp << std::endl;
    // 含义: 在 64 大小的空间中，layout 覆盖了 32 个位置
    // complement 描述了剩余 32 个位置的布局

    std::cout << std::endl;
}

// ============================================================================
// 5. 常见的 MMA 相关 Layout 模式
// ============================================================================

void test_mma_layout_patterns() {
    std::cout << "=== 5. MMA 常见 Layout 模式 ===" << std::endl;
    std::cout << std::endl;

    // ---- Row-Major vs Column-Major ----
    // 在 MMA 中，A 矩阵通常用 Row-Major (TN = Transpose N)
    // B 矩阵也用 Row-Major（但物理上是列主序的 B^T）
    //
    // Row-Major (M, K): stride = (K, 1)
    // Column-Major (M, K): stride = (1, M)

    std::cout << "  Row-Major (4x8):" << std::endl;
    auto row_major = make_layout(make_shape(4, 8));
    // 默认 make_layout 就是 Row-Major: stride = (N, 1) = (8, 1)
    std::cout << "    Layout = " << row_major << std::endl;
    std::cout << "    即 stride = (8, 1): 行内连续，跨行跳 8" << std::endl;

    std::cout << std::endl;
    std::cout << "  Column-Major (4x8):" << std::endl;
    auto col_major = make_layout(make_shape(4, 8), make_stride(1, 4));
    std::cout << "    Layout = " << col_major << std::endl;
    std::cout << "    即 stride = (1, 4): 列内连续，跨列跳 4" << std::endl;

    // ---- MMA 中的线程-值 Layout ----
    // MMA Atom 的核心就是 ThrID -> (M,N) 的映射
    // SM80 的 16x8 MMA 使用 32 个线程 (一个 warp)
    // 每个线程负责输出矩阵的一部分

    std::cout << std::endl;
    std::cout << "  SM80 MMA 线程布局原理:" << std::endl;
    std::cout << "    - 32 个线程排列成 (4,8) 的 2D 网格" << std::endl;
    std::cout << "    - 每个线程处理 M 方向 4 行，N 方向 1 列" << std::endl;
    std::cout << "    - 总共覆盖 16x8 的输出矩阵" << std::endl;
    std::cout << std::endl;

    // 演示 32 线程的 2D 排列
    auto thr_layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                  make_stride(Int<8>{}, Int<1>{}));
    std::cout << "  线程 Layout (4x8):(8,1):" << std::endl;
    std::cout << "    " << thr_layout << std::endl;
    std::cout << "    线程 0 的坐标: (" << 0/8 << "," << 0%8 << ")" << std::endl;
    std::cout << "    线程 15 的坐标: (" << 15/8 << "," << 15%8 << ")" << std::endl;
    std::cout << "    线程 31 的坐标: (" << 31/8 << "," << 31%8 << ")" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 6. make_layout 的默认行为总结
// ============================================================================

void test_make_layout_defaults() {
    std::cout << "=== 6. make_layout 默认行为总结 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  make_layout 的默认 stride 生成规则 (Row-Major):" << std::endl;
    std::cout << std::endl;

    // 1D: stride = (1,)
    auto d1 = make_layout(make_shape(Int<8>{}));
    std::cout << "  make_layout(make_shape(_8))" << std::endl;
    std::cout << "    -> " << d1 << "  (stride = 1)" << std::endl;

    // 2D: stride = (N, 1)
    auto d2 = make_layout(make_shape(Int<4>{}, Int<8>{}));
    std::cout << "  make_layout(make_shape(_4, _8))" << std::endl;
    std::cout << "    -> " << d2 << "  (stride = (8, 1))" << std::endl;

    // 3D: stride = (N*K, K, 1)
    auto d3 = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<4>{}));
    std::cout << "  make_layout(make_shape(_2, _3, _4))" << std::endl;
    std::cout << "    -> " << d3 << "  (stride = (12, 4, 1))" << std::endl;

    // 4D: stride = (N*K*L, K*L, L, 1)
    auto d4 = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<4>{}, Int<5>{}));
    std::cout << "  make_layout(make_shape(_2, _3, _4, _5))" << std::endl;
    std::cout << "    -> " << d4 << "  (stride = (60, 20, 5, 1))" << std::endl;

    std::cout << std::endl;
    std::cout << "  总结：make_layout 默认使用 Row-Major (行优先) stride" << std::endl;
    std::cout << "  stride[i] = product(shape[j], j > i)" << std::endl;
    std::cout << "  即最内维 stride=1，向外依次乘以下一维的大小" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 01: Layout 基础" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_basic_layout();
    test_shape_stride();
    test_nested_layout();
    test_layout_operations();
    test_mma_layout_patterns();
    test_make_layout_defaults();

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 01 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
