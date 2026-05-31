/**
 * ============================================================================
 * CuTe + MMA 教程 03: Copy 基础
 * ============================================================================
 *
 * Copy 是 CuTe 中数据搬运的核心抽象。
 *
 * 核心概念：
 *   - Copy_Atom<Op, T>: 单条指令的拷贝操作
 *   - TiledCopy: 将 Copy_Atom 扩展到更大的数据块
 *   - make_tiled_copy: 创建 TiledCopy 的工厂函数
 *   - partition_S / partition_D: 将 Tensor 按线程分区
 *
 * 编译：make 03_copy_basic
 * 运行：./03_copy_basic
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ============================================================================
// 1. Copy 的核心概念
// ============================================================================
//
// CuTe 的 Copy 操作通过以下层次实现：
//
//   Copy_Atom<Op, T>
//     - Op: 拷贝操作类型 (如 UniversalCopy<uint4>)
//     - T: 数据类型 (如 float, half_t)
//     - 描述单条指令能拷贝多少数据
//
//   TiledCopy
//     - 将 Copy_Atom 扩展到更大的数据块
//     - 定义线程如何分配数据
//     - 使用 make_tiled_copy 创建
//
//   partition_S / partition_D
//     - 将大 Tensor 按线程分区
//     - 返回当前线程负责的 Tensor 片段
//
//   copy(tiled_copy, src, dst)
//     - 执行拷贝操作

void test_copy_concept() {
    std::cout << "=== 1. Copy 核心概念 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  Copy 层次结构:" << std::endl;
    std::cout << "    Copy_Atom<Op, T>: 单条指令的拷贝能力" << std::endl;
    std::cout << "      - Op = UniversalCopy<uint4>: 128-bit 拷贝" << std::endl;
    std::cout << "      - Op = UniversalCopy<uint2>: 64-bit 拷贝" << std::endl;
    std::cout << "      - Op = SM80_CP_ASYNC_CACHEALWAYS: 异步拷贝" << std::endl;
    std::cout << std::endl;
    std::cout << "    TiledCopy: 将 Copy_Atom 扩展到整个 tile" << std::endl;
    std::cout << "      - 定义线程布局 (多少线程)" << std::endl;
    std::cout << "      - 定义值布局 (每线程多少值)" << std::endl;
    std::cout << std::endl;
    std::cout << "    partition: 将 Tensor 按线程分区" << std::endl;
    std::cout << "      - partition_S: 源 Tensor 的分区" << std::endl;
    std::cout << "      - partition_D: 目标 Tensor 的分区" << std::endl;
    std::cout << std::endl;
    std::cout << "    copy: 执行拷贝" << std::endl;
    std::cout << "      - copy(tiled_copy, src, dst)" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 2. 基本 Copy 操作: Global -> Register -> Global
// ============================================================================
// 最简单的 copy 模式：从全局内存加载到寄存器，处理后写回

__global__ void copy_basic_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int N)
{
    // ---- 定义 Copy_Atom ----
    // UniversalCopy<uint4>: 使用 128-bit (16 bytes) 拷贝
    // uint4 = 4 个 unsigned int = 128 bits
    // 对于 float (4 bytes): 128 / 32 = 4 个 float/指令
    using CopyOp = UniversalCopy<uint4>;

    // Copy_Atom<Op, Element>: 将 Op 应用于 Element 类型
    // Copy_Atom<UniversalCopy<uint4>, float> 的含义:
    //   - 使用 128-bit 指令
    //   - 每次拷贝 4 个 float
    //   - 对应 PTX 指令: LDG.E.128 / STG.128
    using Atom = Copy_Atom<CopyOp, float>;

    // ---- 创建 TiledCopy ----
    // make_tiled_copy(atom, thread_layout, value_layout)
    //   atom: Copy_Atom - 每线程的拷贝策略
    //   thread_layout: 256 个线程排列成 (256,)
    //   value_layout: 每线程 4 个值排列成 (4,)
    //
    // 总共搬运: 256 线程 x 4 值/线程 = 1024 个 float = 4KB
    auto tiled_copy = make_tiled_copy(
        Atom{},
        make_layout(make_shape(Int<256>{})),    // 线程布局
        make_layout(make_shape(Int<4>{}))       // 值布局
    );

    // ---- 创建 Tensor ----
    // 注意: 使用编译期常量 Int<1024> 而非运行时 N
    // make_fragment_like 需要编译期静态 shape
    auto src_tensor = make_tensor(src, make_layout(make_shape(Int<1024>{})));
    auto dst_tensor = make_tensor(dst, make_layout(make_shape(Int<1024>{})));

    // ---- 获取线程视图 ----
    // get_thread_slice(threadIdx.x) 返回当前线程负责的数据区域
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

    // ---- 分区 ----
    // partition_S: 将源 Tensor 分区，返回当前线程负责的部分
    // partition_D: 将目标 Tensor 分区
    auto thr_src = thr_copy.partition_S(src_tensor);
    auto thr_dst = thr_copy.partition_D(dst_tensor);

    // ---- 创建寄存器 fragment ----
    // make_fragment_like: 创建与 thr_src 形状相同但存储在寄存器中的 Tensor
    auto frag = make_fragment_like(thr_src);

    // ---- 拷贝: Global -> Register ----
    // copy(tiled_copy, src, dst) 执行拷贝
    // tiled_copy 内部知道如何将 thr_src 的数据拷贝到 frag
    copy(tiled_copy, thr_src, frag);

    // ---- 在寄存器中处理数据 ----
    // CUTE_UNROLL = #pragma unroll，展开循环
    // size(frag) 返回 fragment 中的元素总数
    // frag(i) 访问 fragment 中的第 i 个元素
    CUTE_UNROLL
    for (int i = 0; i < size(frag); ++i) {
        frag(i) = frag(i) * 2.0f;
    }

    // ---- 拷贝: Register -> Global ----
    copy(tiled_copy, frag, thr_dst);
}

void test_copy_basic() {
    std::cout << "=== 2. 基本 Copy 操作 ===" << std::endl;
    std::cout << std::endl;

    const int N = 1024;
    float h_src[1024], h_dst[1024];
    for (int i = 0; i < N; i++) h_src[i] = (float)i;

    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    copy_basic_kernel<<<1, 256>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_dst[i] - h_src[i] * 2.0f) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << "  Copy 结果: " << (correct ? "正确" : "错误") << std::endl;
    std::cout << "  前 8 个元素:" << std::endl;
    std::cout << "    src: ";
    for (int i = 0; i < 8; i++) std::cout << h_src[i] << " ";
    std::cout << std::endl;
    std::cout << "    dst: ";
    for (int i = 0; i < 8; i++) std::cout << h_dst[i] << " ";
    std::cout << std::endl;
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// ============================================================================
// 3. TiledCopy 配置详解
// ============================================================================

void test_tiled_copy_config() {
    std::cout << "=== 3. TiledCopy 配置详解 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  make_tiled_copy(CopyAtom, ThreadLayout, ValueLayout)" << std::endl;
    std::cout << std::endl;

    std::cout << "  CopyAtom 选择:" << std::endl;
    std::cout << "    Copy_Atom<UniversalCopy<uint4>, float>:" << std::endl;
    std::cout << "      - 128-bit 拷贝 (4 个 float/指令)" << std::endl;
    std::cout << "      - 生成 LDG.128 / STG.128 指令" << std::endl;
    std::cout << "    Copy_Atom<UniversalCopy<uint2>, float>:" << std::endl;
    std::cout << "      - 64-bit 拷贝 (2 个 float/指令)" << std::endl;
    std::cout << "    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint4>, float>:" << std::endl;
    std::cout << "      - 异步 128-bit 拷贝 (Global -> Shared)" << std::endl;
    std::cout << std::endl;

    std::cout << "  ThreadLayout 选择:" << std::endl;
    std::cout << "    make_layout(make_shape(Int<256>{}))" << std::endl;
    std::cout << "      - 256 个线程，1D 排列" << std::endl;
    std::cout << "    make_layout(make_shape(Int<32>{}, Int<8>{}))" << std::endl;
    std::cout << "      - 256 个线程，2D 排列 (32x8)" << std::endl;
    std::cout << std::endl;

    std::cout << "  ValueLayout 选择:" << std::endl;
    std::cout << "    make_layout(make_shape(Int<4>{}))" << std::endl;
    std::cout << "      - 每线程 4 个值，1D 排列" << std::endl;
    std::cout << "    make_layout(make_shape(Int<2>{}, Int<2>{}))" << std::endl;
    std::cout << "      - 每线程 4 个值，2D 排列 (2x2)" << std::endl;
    std::cout << std::endl;

    std::cout << "  总数据量 = 线程数 x 每线程值数" << std::endl;
    std::cout << "  例如: 256 线程 x 4 值/线程 = 1024 个 float = 4KB" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 4. 小规模 Copy 示例
// ============================================================================

__global__ void copy_small_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst)
{
    // 4 个线程，每线程 4 个值，总共 16 个 float
    using Atom = Copy_Atom<UniversalCopy<uint4>, float>;

    auto tiled_copy = make_tiled_copy(
        Atom{},
        make_layout(make_shape(Int<4>{})),    // 4 个线程
        make_layout(make_shape(Int<4>{}))     // 每线程 4 个值
    );

    auto src_tensor = make_tensor(src, make_layout(make_shape(Int<16>{})));
    auto dst_tensor = make_tensor(dst, make_layout(make_shape(Int<16>{})));

    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto thr_src = thr_copy.partition_S(src_tensor);
    auto thr_dst = thr_copy.partition_D(dst_tensor);
    auto frag = make_fragment_like(thr_src);

    // 拷贝
    copy(tiled_copy, thr_src, frag);
    copy(tiled_copy, frag, thr_dst);
}

void test_copy_small() {
    std::cout << "=== 4. 小规模 Copy 示例 ===" << std::endl;
    std::cout << std::endl;

    const int N = 16;
    float h_src[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    float h_dst[16] = {0};

    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    copy_small_kernel<<<1, 4>>>(d_src, d_dst);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "  4 线程，每线程 4 值，总共 16 个 float:" << std::endl;
    std::cout << "    线程 0 负责: [0,1,2,3]" << std::endl;
    std::cout << "    线程 1 负责: [4,5,6,7]" << std::endl;
    std::cout << "    线程 2 负责: [8,9,10,11]" << std::endl;
    std::cout << "    线程 3 负责: [12,13,14,15]" << std::endl;
    std::cout << std::endl;
    std::cout << "    src: ";
    for (int i = 0; i < N; i++) std::cout << h_src[i] << " ";
    std::cout << std::endl;
    std::cout << "    dst: ";
    for (int i = 0; i < N; i++) std::cout << h_dst[i] << " ";
    std::cout << std::endl;
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// ============================================================================
// 5. Copy 与 MMA 配合
// ============================================================================

void test_copy_for_mma() {
    std::cout << "=== 5. Copy 与 MMA 配合 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  在 GEMM 中的 Copy 流程:" << std::endl;
    std::cout << std::endl;
    std::cout << "  1. Global -> Shared Memory (使用 TiledCopy + cp.async)" << std::endl;
    std::cout << "     - 创建与 MMA 无关的 TiledCopy" << std::endl;
    std::cout << "     - 使用 SM80_CP_ASYNC_CACHEALWAYS 作为 CopyOp" << std::endl;
    std::cout << std::endl;
    std::cout << "  2. Shared Memory -> Register (使用 TiledCopy + ldmatrix)" << std::endl;
    std::cout << "     - 创建与 MMA 对齐的 TiledCopy" << std::endl;
    std::cout << "     - 使用 make_tiled_copy_A(copy_atom, mma)" << std::endl;
    std::cout << "     - 使用 make_tiled_copy_B(copy_atom, mma)" << std::endl;
    std::cout << std::endl;
    std::cout << "  3. Register -> Global (使用 TiledCopy)" << std::endl;
    std::cout << "     - 使用 make_tiled_copy_C(copy_atom, mma)" << std::endl;
    std::cout << std::endl;
    std::cout << "  make_tiled_copy_A/B/C 的作用:" << std::endl;
    std::cout << "    - 确保 Copy 的线程分区与 MMA 的线程分区一致" << std::endl;
    std::cout << "    - 这样拷贝到寄存器的数据可以直接用于 MMA" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 6. 不同类型的 Copy 操作
// ============================================================================

void test_copy_types() {
    std::cout << "=== 6. 不同类型的 Copy 操作 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  Copy 操作类型 (CopyOp):" << std::endl;
    std::cout << std::endl;
    std::cout << "  1. UniversalCopy<uint4>:" << std::endl;
    std::cout << "     - 128-bit 同步拷贝" << std::endl;
    std::cout << "     - 适合 Global <-> Register" << std::endl;
    std::cout << "     - 生成 LDG.128 / STG.128 指令" << std::endl;
    std::cout << std::endl;
    std::cout << "  2. UniversalCopy<uint2>:" << std::endl;
    std::cout << "     - 64-bit 同步拷贝" << std::endl;
    std::cout << "     - 适合较小的数据块" << std::endl;
    std::cout << std::endl;
    std::cout << "  3. SM80_CP_ASYNC_CACHEALWAYS<uint4>:" << std::endl;
    std::cout << "     - 异步 128-bit 拷贝" << std::endl;
    std::cout << "     - Global -> Shared" << std::endl;
    std::cout << "     - 缓存在 L1 和 L2" << std::endl;
    std::cout << std::endl;
    std::cout << "  4. SM80_CP_ASYNC_CACHEGLOBAL<uint4>:" << std::endl;
    std::cout << "     - 异步 128-bit 拷贝" << std::endl;
    std::cout << "     - Global -> Shared" << std::endl;
    std::cout << "     - 仅缓存在 L2" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 03: Copy 基础" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_copy_concept();
    test_copy_basic();
    test_tiled_copy_config();
    test_copy_small();
    test_copy_for_mma();
    test_copy_types();

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 03 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
