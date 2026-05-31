/**
 * ============================================================================
 * CuTe + MMA 教程 11: 三级流水线 GEMM
 * ============================================================================
 *
 * 完整的异步流水线 GEMM 实现:
 *   1. 3-stage pipeline (三级缓冲)
 *   2. Prologue: 预取前 2 个 tile
 *   3. Main loop: cp_async_wait<K-2> + 计算 + 预取下一个 tile
 *   4. Epilogue: 等待剩余 tile 完成
 *   5. Circular pipe pointer advance
 *
 * 流水线原理:
 *   在计算当前 tile 的同时，异步预取下一个 tile 到 Shared Memory
 *   三级缓冲允许: 计算 tile[n] + 加载 tile[n+1] + 等待 tile[n+2]
 *
 * TN 格式: A(M,K) Row-Major, B(N,K) Row-Major, C(M,N) Col-Major
 *
 * 编译: make 11_software_pipeline
 * 运行: ./11_software_pipeline
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

using MMA_OpType = SM80_16x8x16_F16F16F16F16_TN;
using MMAAtom = MMA_Atom<MMA_OpType>;

constexpr int bM = 128;
constexpr int bN = 128;
constexpr int bK = 64;   // swizzle atom 的倍数

// Pipeline 深度
constexpr int K_PIPE_MAX = 3;

// SharedStorage
template <class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
    cute::ArrayEngine<half_t, cute::cosize_v<SmemLayoutA>> A;
    cute::ArrayEngine<half_t, cute::cosize_v<SmemLayoutB>> B;
};

// ============================================================================
// Pipeline GEMM Kernel
// ============================================================================

template <class ASmemLayout, class BSmemLayout,
          class TiledCopyA, class TiledCopyB,
          class TiledMMA>
__global__ void pipeline_gemm_kernel(
    const half_t* __restrict__ A_ptr,
    const half_t* __restrict__ B_ptr,
    half_t* __restrict__ C_ptr,
    int M, int N, int K)
{
    extern __shared__ char shared_memory[];
    using SmemStorage = SharedStorage<ASmemLayout, BSmemLayout>;
    SmemStorage& smem = *reinterpret_cast<SmemStorage*>(shared_memory);

    // ---- 全局 Tensor ----
    auto dA = make_stride(K, Int<1>{});
    auto dB = make_stride(K, Int<1>{});
    auto dC = make_stride(Int<1>{}, M);

    auto mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), dA);
    auto mB = make_tensor(make_gmem_ptr(B_ptr), make_shape(N, K), dB);
    auto mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});

    auto gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    auto gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    auto gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    // ---- Shared Memory (带 pipeline 维度) ----
    auto sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{});
    auto sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{});

    // ---- G2S Copy ----
    TiledCopyA copyA{};
    TiledCopyB copyB{};

    ThrCopy thr_copy_a = copyA.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);   // (CPY, CPY_M, CPY_K, k_tiles)
    Tensor tAsA = thr_copy_a.partition_D(sA);   // (CPY, CPY_M, CPY_K, PIPE)

    ThrCopy thr_copy_b = copyB.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    // ---- TiledMMA ----
    TiledMMA mma{};
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);

    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = mma.make_fragment_C(tCgC);
    clear(tCrC);

    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));

    // ---- S2R Copy (LDSM) ----
    Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_a;
    Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_b;

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy   s2r_thr_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_a.partition_S(sA);    // (CPY, MMA_M, MMA_K, PIPE)
    Tensor tXrA = s2r_thr_a.retile_D(tCrA);     // (CPY, MMA_M, MMA_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy   s2r_thr_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_b.partition_S(sB);
    Tensor tXrB = s2r_thr_b.retile_D(tCrB);

    // ---- Pipeline 状态 ----
    // k_tile_count: 剩余要处理的K-tile数
    // k_tile_next:   下一个待发起拷贝的K-tile索引
    int k_tile_count = size<3>(tAgA);
    int k_tile_next = 0;

    // 环形缓冲读写指针 (0, 1, 2 三槽循环)
    // 初始: read=0, write=K_PIPE_MAX-1 (最后一个槽)
    int smem_pipe_read  = 0;
    int smem_pipe_write = K_PIPE_MAX - 1;

    auto K_BLOCK_MAX = size<2>(tCrA);

    // ========================================================================
    // Prologue: 预取前K_PIPE_MAX-1个tile (填充流水线)
    // 例: K_PIPE_MAX=3, 预取tile[0]和tile[1]到pipe槽0和1
    // ========================================================================

    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
        // k_pipe: pipe槽编号(目的), k_tile_next: K-tile编号(源)
        cute::copy(copyA, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        cute::copy(copyB, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();  // 提交为一组
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    // PREFETCH: 预取第一个pipe槽的第一个k_block到寄存器
    Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
    Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

    if (K_BLOCK_MAX > 1) {
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();

        cute::copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
        cute::copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
    }

    // ========================================================================
    // Main Loop: while(还有tile未处理完)
    // 条件>-(K_PIPE_MAX-1)确保处理完所有prefetched tile
    // ========================================================================

    CUTE_NO_UNROLL  // while循环不展开
    while (k_tile_count > -(K_PIPE_MAX - 1))
    {
        CUTE_UNROLL  // 内层k_block循环展开
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            // 每个K-tile最后一轮k_block: 等待当前pipe槽就绪
            if (k_block == K_BLOCK_MAX - 1)
            {
                tXsA_p = tXsA(_, _, _, smem_pipe_read);
                tXsB_p = tXsB(_, _, _, smem_pipe_read);

                cp_async_wait<K_PIPE_MAX - 2>();  // 允许至多1个未完成组
                __syncthreads();                    // 等待所有线程
            }

            // S2R: 提前取下一个k_block (寄存器流水线)
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
            cute::copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
            cute::copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));

            // G2S: 每个K-tile第一轮k_block时发起下一tile的cp.async
            if (k_block == 0)
            {
                cute::copy(copyA, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                cute::copy(copyB, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                cp_async_fence();

                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile_next; }

                // 环形推进: write追上read, read前进一步
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
            }

            // MMA: Tensor Core计算当前k_block
            cute::gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    // ========================================================================
    // Epilogue: 写回结果
    // ========================================================================
    cute::copy(tCrC, tCgC);
}

// ============================================================================
// CPU GEMM + 验证
// ============================================================================

void cpu_gemm(const half_t* A, const half_t* B, half_t* C,
              int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += static_cast<float>(A[m * K + k]) * static_cast<float>(B[n * K + k]);
            C[m + n * M] = static_cast<half_t>(sum);
        }
}

bool verify_gemm(const half_t* C_gpu, const half_t* A, const half_t* B,
                 int M, int N, int K) {
    half_t* C_ref = new half_t[M * N];
    cpu_gemm(A, B, C_ref, M, N, K);

    float max_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < M * N; i++) {
        float g = static_cast<float>(C_gpu[i]);
        float c = static_cast<float>(C_ref[i]);
        float err = fabsf(g - c);
        max_err = fmaxf(max_err, err);
        float ref = fmaxf(fabsf(c), 1.0f);
        if (err / ref > 0.01f) err_count++;
    }

    std::cout << "  最大误差: " << max_err << std::endl;
    std::cout << "  " << (err_count == 0 ? "验证通过!" : "验证失败!") << std::endl;

    delete[] C_ref;
    return err_count == 0;
}

// ============================================================================
// 测试
// ============================================================================

void test_pipeline_gemm(int M, int N, int K) {
    std::cout << "=== 测试: 三级流水线 GEMM " << M << "x" << N << "x" << K << " ===" << std::endl;

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

    // TiledMMA: 2x2 atoms, 32x32x16 tiling for LDSM
    auto tiled_mma = make_tiled_mma(MMAAtom{},
                                    Layout<Shape<_2, _2>>{},
                                    Tile<_32, _32, _16>{});

    // Swizzled Shared Memory (带 pipeline 维度)
    auto swizzle_atom = composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<_8, Shape<_8, _8>>,
               Stride<_8, Stride<_1, _64>>>{}
    );
    auto sA = tile_to_shape(swizzle_atom, make_shape(Int<bM>{}, Int<bK>{}, Int<K_PIPE_MAX>{}));
    auto sB = tile_to_shape(swizzle_atom, make_shape(Int<bN>{}, Int<bK>{}, Int<K_PIPE_MAX>{}));

    // G2S Copy
    using G2SAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>;
    auto copyA = make_tiled_copy(G2SAtom{},
                                 Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                 Layout<Shape<_1, _8>>{});
    auto copyB = make_tiled_copy(G2SAtom{},
                                 Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                                 Layout<Shape<_1, _8>>{});

    int smem_size = int(sizeof(SharedStorage<decltype(sA), decltype(sB)>));

    dim3 grid(ceil_div(M, bM), ceil_div(N, bN));
    dim3 block(size(tiled_mma));

    auto kernel = pipeline_gemm_kernel<decltype(sA), decltype(sB),
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
// 流水线说明
// ============================================================================

void print_pipeline_info() {
    std::cout << "=== 三级流水线说明 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  Pipeline 状态:" << std::endl;
    std::cout << "    K_PIPE_MAX = 3 (三级缓冲)" << std::endl;
    std::cout << "    smem_pipe_read:  当前读取的 pipe 索引" << std::endl;
    std::cout << "    smem_pipe_write: 当前写入的 pipe 索引" << std::endl;
    std::cout << std::endl;

    std::cout << "  Prologue (预取):" << std::endl;
    std::cout << "    for k_pipe in [0, K_PIPE_MAX-1):" << std::endl;
    std::cout << "      issue cp.async for tile k_pipe" << std::endl;
    std::cout << "      cp_async_fence()" << std::endl;
    std::cout << std::endl;

    std::cout << "  Main Loop:" << std::endl;
    std::cout << "    while (more tiles):" << std::endl;
    std::cout << "      for k_block in [0, K_BLOCK_MAX):" << std::endl;
    std::cout << "        if (last k_block):" << std::endl;
    std::cout << "          cp_async_wait<K_PIPE_MAX-2>()" << std::endl;
    std::cout << "          __syncthreads()" << std::endl;
    std::cout << "        S2R: prefetch next k_block" << std::endl;
    std::cout << "        if (first k_block):" << std::endl;
    std::cout << "          issue cp.async for next tile" << std::endl;
    std::cout << "          advance pipe pointers" << std::endl;
    std::cout << "        MMA: gemm(k_block)" << std::endl;
    std::cout << std::endl;

    std::cout << "  时间线 (三级缓冲):" << std::endl;
    std::cout << "    时间 0: 加载 tile[0], 加载 tile[1]" << std::endl;
    std::cout << "    时间 1: 计算 tile[0], 加载 tile[2]" << std::endl;
    std::cout << "    时间 2: 计算 tile[1], 加载 tile[3]" << std::endl;
    std::cout << "    ..." << std::endl;
    std::cout << "    计算与加载始终重叠 1 个 tile" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 11: 三级流水线 GEMM" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    print_pipeline_info();
    test_pipeline_gemm(128, 128, 256);
    test_pipeline_gemm(256, 256, 256);

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 11 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
