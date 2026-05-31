/**
 * ============================================================================
 * CuTe + MMA 教程 07: TiledMMA
 * ============================================================================
 *
 * TiledMMA 将单个 MMA_Atom 扩展到更大的 tile。
 *
 * 核心概念：
 *   - make_tiled_mma: 创建 TiledMMA 的工厂函数
 *   - tile_shape / tile_size: 获取 tile 的形状
 *   - partition_A/B/C: 将大矩阵分给每个线程
 *   - partition_fragment_A/B: 创建寄存器 fragment
 *   - make_fragment_C: 创建 C 的寄存器 fragment
 *
 * 编译：make 07_tiled_mma
 * 运行：./07_tiled_mma
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>

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
// 1. TiledMMA 基础
// ============================================================================

void test_tiled_mma_basic() {
    std::cout << "=== 1. TiledMMA 基础 ===" << std::endl;
    std::cout << std::endl;

    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;

    // 默认 TiledMMA
    auto tiled_mma = make_tiled_mma(MMA{});

    std::cout << "  默认 TiledMMA (1 个 Atom):" << std::endl;
    std::cout << "    Tile Shape = " << tile_shape(tiled_mma) << std::endl;
    std::cout << "    Tile M = " << tile_size<0>(tiled_mma) << std::endl;
    std::cout << "    Tile N = " << tile_size<1>(tiled_mma) << std::endl;
    std::cout << "    Tile K = " << tile_size<2>(tiled_mma) << std::endl;
    std::cout << "    线程数 = " << size(tiled_mma) << std::endl;
    std::cout << std::endl;

    // 扩展 TiledMMA
    auto tiled_mma_2x2 = make_tiled_mma(MMA{},
                                         Layout<Shape<_2, _2, _1>>{});

    std::cout << "  扩展 TiledMMA (2x2 Atoms):" << std::endl;
    std::cout << "    Tile Shape = " << tile_shape(tiled_mma_2x2) << std::endl;
    std::cout << "    Tile M = " << tile_size<0>(tiled_mma_2x2) << std::endl;
    std::cout << "    Tile N = " << tile_size<1>(tiled_mma_2x2) << std::endl;
    std::cout << "    Tile K = " << tile_size<2>(tiled_mma_2x2) << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 2. TiledMMA 的 partition 操作
// ============================================================================

void test_partition() {
    std::cout << "=== 2. TiledMMA 的 partition 操作 ===" << std::endl;
    std::cout << std::endl;

    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;
    auto tiled_mma = make_tiled_mma(MMA{});

    std::cout << "  MMA 形状 (M,N,K): " << tile_shape(tiled_mma) << std::endl;
    std::cout << std::endl;

    // 说明 partition 的含义
    std::cout << "  partition 操作说明:" << std::endl;
    std::cout << std::endl;
    std::cout << "  partition_A(tensor):" << std::endl;
    std::cout << "    - 将 A 矩阵 (M x K) 按 MMA 的线程布局分区" << std::endl;
    std::cout << "    - 返回每个线程负责的 A 片段" << std::endl;
    std::cout << "    - 形状: (MMA_M, MMA_K) 每线程" << std::endl;
    std::cout << std::endl;
    std::cout << "  partition_B(tensor):" << std::endl;
    std::cout << "    - 将 B 矩阵 (N x K) 按 MMA 的线程布局分区" << std::endl;
    std::cout << "    - 返回每个线程负责的 B 片段" << std::endl;
    std::cout << "    - 形状: (MMA_N, MMA_K) 每线程" << std::endl;
    std::cout << std::endl;
    std::cout << "  partition_C(tensor):" << std::endl;
    std::cout << "    - 将 C 矩阵 (M x N) 按 MMA 的线程布局分区" << std::endl;
    std::cout << "    - 返回每个线程负责的 C 片段" << std::endl;
    std::cout << "    - 形状: (MMA_M, MMA_N) 每线程" << std::endl;
    std::cout << std::endl;
    std::cout << "  partition_fragment_A/B:" << std::endl;
    std::cout << "    - 创建 A/B 的寄存器 fragment" << std::endl;
    std::cout << "    - 用于存储从 Shared Memory 加载的数据" << std::endl;
    std::cout << std::endl;
    std::cout << "  make_fragment_C:" << std::endl;
    std::cout << "    - 创建 C 的寄存器 fragment (累加器)" << std::endl;
    std::cout << "    - 形状与 partition_C 的结果匹配" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 3. 完整的 MMA Kernel
// ============================================================================
// TN 格式: A 是 (M, K), B 是 (N, K), C 是 (M, N)

__global__ void tiled_mma_kernel(
    const half* __restrict__ A_ptr,
    const half* __restrict__ B_ptr,
    float* __restrict__ C_ptr)
{
    using MMA = MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>;
    auto tiled_mma = make_tiled_mma(MMA{});

    int tid = threadIdx.x;

    // 创建全局内存 Tensor
    // A: (M, K) = (16, 8)
    auto A = make_tensor(make_gmem_ptr(A_ptr),
                         make_layout(make_shape(Int<16>{}, Int<8>{})));
    // B: (N, K) = (8, 8) - TN 格式
    auto B = make_tensor(make_gmem_ptr(B_ptr),
                         make_layout(make_shape(Int<8>{}, Int<8>{})));
    // C: (M, N) = (16, 8)
    auto C = make_tensor(make_gmem_ptr(C_ptr),
                         make_layout(make_shape(Int<16>{}, Int<8>{})));

    // 获取线程视图
    auto thr_mma = tiled_mma.get_slice(tid);

    // 创建寄存器 fragment
    // partition_fragment_A: 使用共享内存布局作为模板创建寄存器 fragment
    // 这里我们直接使用全局内存 Tensor 作为模板
    auto frag_A = thr_mma.partition_fragment_A(A);
    auto frag_B = thr_mma.partition_fragment_B(B);
    auto part_C = thr_mma.partition_C(C);
    auto frag_C = tiled_mma.make_fragment_C(part_C);

    // 初始化 C fragment 为 0
    clear(frag_C);

    // 从全局内存加载 A 和 B 到寄存器
    // 注意: 这里直接 copy，实际 GEMM 中应该从 Shared Memory 加载
    auto part_A = thr_mma.partition_A(A);
    auto part_B = thr_mma.partition_B(B);
    cute::copy(part_A, frag_A);
    cute::copy(part_B, frag_B);

    // 执行 MMA: C = A * B + C
    cute::gemm(tiled_mma, frag_A, frag_B, frag_C);

    // 将结果写回全局内存
    cute::copy(frag_C, part_C);
}

void test_tiled_mma_kernel() {
    std::cout << "=== 3. TiledMMA Kernel ===" << std::endl;
    std::cout << std::endl;

    const int M = 16, N = 8, K = 8;

    half h_A[16 * 8], h_B[8 * 8];
    float h_C[16 * 8];

    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < N * K; i++) h_B[i] = __float2half(1.0f);
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    tiled_mma_kernel<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "  MMA 结果 (A 全 1, B 全 1, K=8):" << std::endl;
    std::cout << "    期望: 每个元素 = 8.0" << std::endl;
    std::cout << "    实际:" << std::endl;
    for (int m = 0; m < M; m++) {
        std::cout << "      ";
        for (int n = 0; n < N; n++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(1) << h_C[m * N + n];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ============================================================================
// 4. 使用说明
// ============================================================================

void test_usage_info() {
    std::cout << "=== 4. TiledMMA 使用说明 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  TN 格式说明:" << std::endl;
    std::cout << "    A: (M, K) - Row-Major" << std::endl;
    std::cout << "    B: (N, K) - Row-Major (注意: 不是 K x N)" << std::endl;
    std::cout << "    C: (M, N) - Row-Major" << std::endl;
    std::cout << std::endl;

    std::cout << "  在实际 GEMM 中的使用流程:" << std::endl;
    std::cout << "    1. 定义 MMA 和 TiledMMA" << std::endl;
    std::cout << "    2. 创建 Shared Memory Tensor" << std::endl;
    std::cout << "    3. 使用 TiledCopy 从 Global 加载到 Shared" << std::endl;
    std::cout << "    4. 使用 partition_fragment 创建寄存器" << std::endl;
    std::cout << "    5. 使用 make_tiled_copy_A/B 从 Shared 加载到寄存器" << std::endl;
    std::cout << "    6. 执行 cute::gemm" << std::endl;
    std::cout << "    7. 写回结果" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 07: TiledMMA" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_tiled_mma_basic();
    test_partition();
    test_tiled_mma_kernel();
    test_usage_info();

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 07 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
