/**
 * ============================================================================
 * CuTe + MMA 教程 10: Shared Memory GEMM
 * ============================================================================
 *
 * 在教程 08 (基础 GEMM) 的基础上加入 Shared Memory 层:
 *   1. make_smem_ptr 创建 Shared Memory Tensor
 *   2. Swizzle 布局: Swizzle<3,3,3> 避免 bank conflict
 *   3. tile_to_shape 将 swizzle atom 扩展到 tile 大小
 *   4. G2S: cp.async Global→Shared
 *   5. S2R: make_tiled_copy_A/B 从 Shared 加载到寄存器
 *   6. K-loop: 每个 K-tile 先 G2S，再 S2R，再 MMA
 *
 * TN 格式: A(M,K) Row-Major, B(N,K) Row-Major, C(M,N) Col-Major
 *
 * 编译: make 10_shared_mem_gemm
 * 运行: ./10_shared_mem_gemm
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>

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
// GEMM 配置
// ============================================================================

// MMA: 16x8x16, FP16 input, FP16 accumulate
using MMA_OpType = SM80_16x8x16_F16F16F16F16_TN;
using MMAAtom = MMA_Atom<MMA_OpType>;

// Block tile 大小
// bK 必须是 swizzle atom K 维度 (64) 的倍数
constexpr int bM = 128;
constexpr int bN = 128;
constexpr int bK = 64;

// SharedStorage
template <class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
    cute::ArrayEngine<half_t, cute::cosize_v<SmemLayoutA>> A;
    cute::ArrayEngine<half_t, cute::cosize_v<SmemLayoutB>> B;
};

// ============================================================================
// GEMM Kernel (模板参数全部编译期确定)
// ============================================================================
//
// 模板参数:
//   ASmemLayout, BSmemLayout: swizzled shared memory布局
//   TiledCopyA, TiledCopyB:    G2S cp.async拷贝策略
//   TiledMMA:                  MMA tile策略

template <class ASmemLayout, class BSmemLayout,
          class TiledCopyA, class TiledCopyB,
          class TiledMMA>
__global__ void shared_mem_gemm_kernel(
    const half_t* __restrict__ A_ptr,  // [M x K] Row-Major
    const half_t* __restrict__ B_ptr,  // [N x K] Row-Major (TN格式)
    half_t* __restrict__ C_ptr,        // [M x N] Col-Major 输出
    int M, int N, int K)               // 矩阵维度 (运行时)
{
    // ---- Shared Memory ----
    extern __shared__ char shared_memory[];
    using SmemStorage = SharedStorage<ASmemLayout, BSmemLayout>;
    SmemStorage& smem = *reinterpret_cast<SmemStorage*>(shared_memory);

    // ---- 全局 Tensor ----
    // stride=(K,1): Row-Major;  stride=(1,M): Col-Major
    auto dA = make_stride(K, Int<1>{});
    auto dB = make_stride(K, Int<1>{});
    auto dC = make_stride(Int<1>{}, M);

    auto mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), dA);
    auto mB = make_tensor(make_gmem_ptr(B_ptr), make_shape(N, K), dB);
    auto mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), dC);

    // ---- Block坐标: 提取当前CTA负责的子矩阵 ----
    // local_tile(tensor, tiler, coord, Step): 从大矩阵中切出一块
    // Step<_1,X,_1>: A沿M和K方向取, N方向跳过(由blockIdx.y不参与)
    // Step<X,_1,_1>: B沿N和K方向取, M方向跳过
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});

    auto gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    auto gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    auto gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    // ---- Shared Memory Tensor ----
    auto sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{});
    auto sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{});

    // ---- G2S Copy (Global → Shared, cp.async) ----
    TiledCopyA copyA{};
    TiledCopyB copyB{};

    ThrCopy thr_copy_a = copyA.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);   // (CPY, CPY_M, CPY_K, k_tiles)
    Tensor tAsA = thr_copy_a.partition_D(sA);   // (CPY, CPY_M, CPY_K)

    ThrCopy thr_copy_b = copyB.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);   // (CPY, CPY_N, CPY_K, k_tiles)
    Tensor tBsB = thr_copy_b.partition_D(sB);   // (CPY, CPY_N, CPY_K)

    // ---- TiledMMA ----
    TiledMMA tiled_mma{};
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);

    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = tiled_mma.make_fragment_C(tCgC);
    clear(tCrC);

    // A/B register fragments
    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);

    // ---- S2R Copy (Shared→Register, 用LDSM指令) ----
    // LDSM = ldmatrix, 从Shared Memory加载矩阵片段到寄存器
    // make_tiled_copy_A/B: 与MMA的线程/寄存器布局对齐的copy策略
    Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_a;
    Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_b;

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, tiled_mma);
    ThrCopy   s2r_thr_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_a.partition_S(sA);    // (CPY, MMA_M, MMA_K) 源视图
    Tensor tXrA = s2r_thr_a.retile_D(tCrA);     // (CPY, MMA_M, MMA_K) retile到fragment布局

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, tiled_mma);
    ThrCopy   s2r_thr_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_b.partition_S(sB);    // (CPY, MMA_N, MMA_K)
    Tensor tXrB = s2r_thr_b.retile_D(tCrB);     // (CPY, MMA_N, MMA_K)

    // ---- K-loop: 遍历所有K方向的tile ----
    // k_tile_count: K方向被切成几个bK大小的tile
    int k_tile_count = size<3>(tAgA);

    for (int k = 0; k < k_tile_count; ++k) {
        // Step 1: G2S — cp.async 异步拷贝 Global→Shared
        // k是K-tile索引, 用(_, _, _, k)取出第k个tile
        cute::copy(copyA, tAgA(_, _, _, k), tAsA);
        cute::copy(copyB, tBgB(_, _, _, k), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // Step 2: S2R - 从 Shared 加载到寄存器
        // 使用 MMA 的 K 维度进行循环
        auto K_BLOCK_MAX = size<2>(tCrA);
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            cute::copy(s2r_atom_a, tXsA(_, _, k_block), tXrA(_, _, k_block));
            cute::copy(s2r_atom_b, tXsB(_, _, k_block), tXrB(_, _, k_block));

            // Step 3: MMA - Tensor Core 矩阵乘法
            cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }

        __syncthreads();
    }

    // ---- 写回结果 ----
    cute::copy(tCrC, tCgC);
}

// ============================================================================
// CPU GEMM (验证用)
// ============================================================================

void cpu_gemm(const half_t* A, const half_t* B, half_t* C,
              int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += static_cast<float>(A[m * K + k]) * static_cast<float>(B[n * K + k]);
            }
            // Col-Major: C[m + n*M]
            C[m + n * M] = static_cast<half_t>(sum);
        }
    }
}

// ============================================================================
// 验证函数
// ============================================================================

bool verify_gemm(const half_t* C_gpu, const half_t* A, const half_t* B,
                 int M, int N, int K) {
    half_t* C_ref = new half_t[M * N];
    cpu_gemm(A, B, C_ref, M, N, K);

    float max_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < M * N; i++) {
        float gpu_val = static_cast<float>(C_gpu[i]);
        float cpu_val = static_cast<float>(C_ref[i]);
        float err = fabsf(gpu_val - cpu_val);
        max_err = fmaxf(max_err, err);
        float ref = fmaxf(fabsf(cpu_val), 1.0f);
        if (err / ref > 0.01f) err_count++;
    }

    std::cout << "  最大误差: " << max_err << std::endl;
    if (err_count == 0) {
        std::cout << "  验证通过!" << std::endl;
    } else {
        std::cout << "  验证失败! 误差元素: " << err_count << std::endl;
    }

    delete[] C_ref;
    return err_count == 0;
}

// ============================================================================
// 测试用例
// ============================================================================

void test_gemm(int M, int N, int K) {
    std::cout << "=== 测试: GEMM " << M << "x" << N << "x" << K << " ===" << std::endl;

    half_t* h_A = new half_t[M * K];
    half_t* h_B = new half_t[N * K];
    half_t* h_C = new half_t[M * N];

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<half_t>((float)(rand() % 10) / 10.0f);
    for (int i = 0; i < N * K; i++) h_B[i] = static_cast<half_t>((float)(rand() % 10) / 10.0f);
    for (int i = 0; i < M * N; i++) h_C[i] = static_cast<half_t>(0.0f);

    half_t *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half_t)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * K * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(half_t), cudaMemcpyHostToDevice));

    // TiledMMA: 2x2 MMA atoms, 32x32x16 tiling for LDSM compatibility
    auto tiled_mma = make_tiled_mma(MMAAtom{},
                                    Layout<Shape<_2, _2>>{},
                                    Tile<_32, _32, _16>{});

    // Swizzled Shared Memory 布局
    // Swizzle<3,3,3> 打乱行访问模式，避免 bank conflict
    auto swizzle_atom = composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<_8, Shape<_8, _8>>,
               Stride<_8, Stride<_1, _64>>>{}
    );
    auto sA = tile_to_shape(swizzle_atom, make_shape(Int<bM>{}, Int<bK>{}));
    auto sB = tile_to_shape(swizzle_atom, make_shape(Int<bN>{}, Int<bK>{}));

    // G2S Copy (cp.async, 128-bit)
    using G2SAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>;
    auto copyA = make_tiled_copy(G2SAtom{},
                                 Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                 Layout<Shape<_1, _8>>{});
    auto copyB = make_tiled_copy(G2SAtom{},
                                 Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                 Layout<Shape<_1, _8>>{});

    // Shared Memory 大小
    int smem_size = int(sizeof(SharedStorage<decltype(sA), decltype(sB)>));

    dim3 grid(ceil_div(M, bM), ceil_div(N, bN));
    dim3 block(size(tiled_mma));

    auto kernel = shared_mem_gemm_kernel<decltype(sA), decltype(sB),
                                          decltype(copyA), decltype(copyB),
                                          decltype(tiled_mma)>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    kernel<<<grid, block, smem_size>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(half_t), cudaMemcpyDeviceToHost));

    verify_gemm(h_C, h_A, h_B, M, N, K);
    std::cout << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ============================================================================
// 架构说明
// ============================================================================

void print_architecture() {
    std::cout << "=== Shared Memory GEMM 架构 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  数据流:" << std::endl;
    std::cout << "    Global Memory (A, B)" << std::endl;
    std::cout << "         |" << std::endl;
    std::cout << "         | cp.async (异步, 不经过寄存器)" << std::endl;
    std::cout << "         v" << std::endl;
    std::cout << "    Shared Memory (sA, sB) [Swizzled]" << std::endl;
    std::cout << "         |" << std::endl;
    std::cout << "         | LDSM (ldmatrix 指令)" << std::endl;
    std::cout << "         v" << std::endl;
    std::cout << "    Registers (frag_A, frag_B)" << std::endl;
    std::cout << "         |" << std::endl;
    std::cout << "         | MMA (Tensor Core)" << std::endl;
    std::cout << "         v" << std::endl;
    std::cout << "    Registers (frag_C)" << std::endl;
    std::cout << "         |" << std::endl;
    std::cout << "         | st.global" << std::endl;
    std::cout << "         v" << std::endl;
    std::cout << "    Global Memory (C)" << std::endl;
    std::cout << std::endl;

    std::cout << "  Swizzle 布局:" << std::endl;
    std::cout << "    Swizzle<3,3,3> 打乱行访问模式" << std::endl;
    std::cout << "    避免不同 thread 访问同一 bank" << std::endl;
    std::cout << std::endl;

    std::cout << "  LDSM 指令:" << std::endl;
    std::cout << "    SM75_U32x4_LDSM_N: ldmatrix 指令" << std::endl;
    std::cout << "    从 Shared Memory 加载矩阵片段到寄存器" << std::endl;
    std::cout << "    与 MMA 的寄存器布局天然对齐" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 10: Shared Memory GEMM" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    print_architecture();
    test_gemm(128, 128, 64);
    test_gemm(256, 256, 128);
    test_gemm(128, 128, 256);

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 10 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
