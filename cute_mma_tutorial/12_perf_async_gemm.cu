/**
 * ============================================================================
 * CuTe + MMA 教程 12: 完整性能测试 GEMM
 * ============================================================================
 *
 * 可 NCU profile 的大矩阵 GEMM，展示所有优化技术:
 *   1. Block-level tiling: (bM, bN, bK) tile 覆盖整个矩阵
 *   2. 完整异步流水线: cp.async G2S + LDSM S2R + MMA
 *   3. Swizzled shared memory
 *   4. 支持任意 M, N, K 大小 (运行时参数)
 *   5. CPU 验证 + 性能统计 (GFLOPS)
 *
 * TN 格式: A(M,K) Row-Major, B(N,K) Row-Major, C(M,N) Col-Major
 *
 * 编译: make 12_perf_async_gemm
 * 运行: ./12_perf_async_gemm [M] [N] [K]
 * NCU:  make ncu_12_perf_async_gemm
 */

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
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
constexpr int bK = 64;
constexpr int K_PIPE_MAX = 3;

// ============================================================================
// SharedStorage
// ============================================================================

template <class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
    cute::ArrayEngine<half_t, cute::cosize_v<SmemLayoutA>> A;
    cute::ArrayEngine<half_t, cute::cosize_v<SmemLayoutB>> B;
};

// ============================================================================
// GEMM Kernel (全部配置编译期模板化)
// ============================================================================
//
// 模板参数:
//   ProblemShape: 编译期shape tuple (M,N,K)
//   CtaTiler:     CTA tile大小 (Int<bM>,Int<bN>,Int<bK>)
//   ASmemLayout, BSmemLayout: swizzled+pipelined SMEM布局
//   TiledCopyA/B: G2S cp.async拷贝策略
//   TiledMMA:     MMA tile策略 (2x2 atoms, 32x32x16)

template <class ProblemShape, class CtaTiler,
          class ASmemLayout, class BSmemLayout,
          class TiledCopyA, class TiledCopyB,
          class TiledMMA>
__global__ void
__launch_bounds__(decltype(size(TiledMMA{}))::value)  // 静态线程数限制
gemm_kernel(
    ProblemShape shape_MNK,      // tuple(M,N,K), get<0/1/2>获取各维
    CtaTiler cta_tiler,          // CTA tile (编译期值, Int<>包装)
    half_t const* A_ptr,         // [M x K] Row-Major
    half_t const* B_ptr,         // [N x K] Row-Major (TN)
    half_t* C_ptr,               // [M x N] Col-Major 输出
    ASmemLayout sA_layout,       // A的swizzle SMEM布局 (值传递)
    BSmemLayout sB_layout,       // B的swizzle SMEM布局
    TiledCopyA copyA,            // A的G2S拷贝 (值传递)
    TiledCopyB copyB,            // B的G2S拷贝
    TiledMMA mma)                // MMA策略 (值传递)
{
    using namespace cute;

    extern __shared__ char shared_memory[];
    using SmemStorage = SharedStorage<ASmemLayout, BSmemLayout>;
    SmemStorage& smem = *reinterpret_cast<SmemStorage*>(shared_memory);

    // ---- 全局 Tensor ----
    // get<2>(shape_MNK): 从tuple(M,N,K)中取K, stride=(K,1) Row-Major
    // get<0>(shape_MNK): 取M, Col-Major stride=(1,M)
    auto dA = make_stride(get<2>(shape_MNK), Int<1>{});
    auto dB = make_stride(get<2>(shape_MNK), Int<1>{});
    auto dC = make_stride(Int<1>{}, get<0>(shape_MNK));

    // select<0,2>(M,N,K) -> (M,K), select<1,2> -> (N,K), select<0,1> -> (M,N)
    Tensor mA = make_tensor(make_gmem_ptr(A_ptr), select<0,2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B_ptr), select<1,2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C_ptr), select<0,1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    // ---- Shared Memory ----
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

    // ---- G2S Copy ----
    ThrCopy thr_copy_a = copyA.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);

    ThrCopy thr_copy_b = copyB.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    // ---- TiledMMA ----
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = mma.make_fragment_C(tCgC);
    clear(tCrC);

    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));

    // ---- S2R Copy (LDSM: Shared→Register) ----
    // make_tiled_copy_A/B: 对齐MMA布局, retile_D推土到fragment形状
    Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_a;
    Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_b;

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy   s2r_thr_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_a.partition_S(sA);    // (CPY, MMA_M, MMA_K, PIPE)
    Tensor tXrA = s2r_thr_a.retile_D(tCrA);     // retile到fragment布局

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy   s2r_thr_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_b.partition_S(sB);
    Tensor tXrB = s2r_thr_b.retile_D(tCrB);

    // ---- Pipeline 状态 ----
    int k_tile_count = size<3>(tAgA);  // 总K-tile数
    int k_tile_next = 0;                // 下一个待发起的K-tile索引

    // ========================================================================
    // Prologue: 预取K_PIPE_MAX-1个tile, 填充流水线
    // ========================================================================

    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
        cute::copy(copyA, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        cute::copy(copyB, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    auto K_BLOCK_MAX = size<2>(tCrA);

    int smem_pipe_read  = 0;
    int smem_pipe_write = K_PIPE_MAX - 1;

    Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
    Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();

        cute::copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
        cute::copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
    }

    // ========================================================================
    // Main Loop: 流水线执行 (G2S + S2R + MMA 重叠)
    // ========================================================================

    CUTE_NO_UNROLL  // while不展开
    while (k_tile_count > -(K_PIPE_MAX - 1))
    {
        CUTE_UNROLL  // k_block循环展开
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            // 每个K-tile末尾: 等待pipe槽数据就绪
            if (k_block == K_BLOCK_MAX - 1)
            {
                tXsA_p = tXsA(_, _, _, smem_pipe_read);
                tXsB_p = tXsB(_, _, _, smem_pipe_read);

                cp_async_wait<K_PIPE_MAX - 2>();  // 允许至多1个未完成组
                __syncthreads() ;                   // warp间同步
            }

            // S2R预取: 提前加载下一个k_block到寄存器
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
            cute::copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
            cute::copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));

            // G2S: 每个K-tile开头发起下一个tile的cp.async
            if (k_block == 0)
            {
                cute::copy(copyA, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                cute::copy(copyB, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                cp_async_fence();

                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile_next; }

                // 环形指针: write接替read, read前进
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX - 1) ? 0 : smem_pipe_read + 1;
            }

            // MMA: Tensor Core矩阵乘加
            cute::gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    // ========================================================================
    // Epilogue
    // ========================================================================
    cute::copy(tCrC, tCgC);
}

// ============================================================================
// Host GEMM launcher
// ============================================================================

// Host端GEMM启动函数
// M,N,K: 矩阵维度, stream: CUDA流(默认0)
void gemm_tn(int M, int N, int K,
             half_t const* A, half_t const* B, half_t* C,
             cudaStream_t stream = 0)
{
    using namespace cute;

    // prob_shape: (M,N,K) 用于select<0,2>提取shape子集
    auto prob_shape = make_shape(M, N, K);
    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});

    // Swizzled Shared Memory
    auto swizzle_atom = composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<_8, Shape<_8, _8>>,
               Stride<_8, Stride<_1, _64>>>{}
    );
    auto sA = tile_to_shape(swizzle_atom, make_shape(Int<bM>{}, Int<bK>{}, Int<K_PIPE_MAX>{}));
    auto sB = tile_to_shape(swizzle_atom, make_shape(Int<bN>{}, Int<bK>{}, Int<K_PIPE_MAX>{}));

    // G2S Copy (cp.async)
    TiledCopy copyA = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}
    );
    TiledCopy copyB = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}
    );

    // TiledMMA: 2x2 atoms, 32x32x16 tiling
    TiledMMA mma = make_tiled_mma(MMAAtom{},
                                  Layout<Shape<_2, _2>>{},
                                  Tile<_32, _32, _16>{});

    int smem_size = int(sizeof(SharedStorage<decltype(sA), decltype(sB)>));

    dim3 dimGrid(size(ceil_div(M, Int<bM>{})),
                 size(ceil_div(N, Int<bN>{})));
    dim3 dimBlock(size(mma));

    auto kernel = gemm_kernel<
        decltype(prob_shape), decltype(cta_tiler),
        decltype(sA), decltype(sB),
        decltype(copyA), decltype(copyB),
        decltype(mma)>;

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    kernel<<<dimGrid, dimBlock, smem_size, stream>>>(
        prob_shape, cta_tiler,
        A, B, C,
        sA, sB,
        copyA, copyB,
        mma);
}

// ============================================================================
// CPU GEMM
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

// ============================================================================
// 验证
// ============================================================================

bool verify(const half_t* gpu_C, const half_t* cpu_C, int M, int N) {
    float max_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < M * N; i++) {
        float g = static_cast<float>(gpu_C[i]);
        float c = static_cast<float>(cpu_C[i]);
        float err = fabsf(g - c);
        max_err = fmaxf(max_err, err);
        float ref = fmaxf(fabsf(c), 1.0f);
        if (err / ref > 0.05f && err > 0.01f) err_count++;
    }
    std::cout << "  最大绝对误差: " << max_err << std::endl;
    if (err_count == 0) {
        std::cout << "  验证通过!" << std::endl;
        return true;
    } else {
        std::cout << "  验证失败! 误差元素: " << err_count << " / " << M * N << std::endl;
        return false;
    }
}

// ============================================================================
// 计时工具
// ============================================================================

struct CudaTimer {
    cudaEvent_t start_evt, stop_evt;
    CudaTimer() {
        cudaEventCreate(&start_evt);
        cudaEventCreate(&stop_evt);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }
    void start(cudaStream_t s = 0) { cudaEventRecord(start_evt, s); }
    void stop(cudaStream_t s = 0)  { cudaEventRecord(stop_evt, s); }
    float elapsed_ms() {
        cudaEventSynchronize(stop_evt);
        float ms;
        cudaEventElapsedTime(&ms, start_evt, stop_evt);
        return ms;
    }
};

// ============================================================================
// 性能测试
// ============================================================================

void perf_test(int M, int N, int K, bool do_verify = true) {
    std::cout << "=== 性能测试: GEMM " << M << "x" << N << "x" << K << " ===" << std::endl;

    half_t* h_A = new half_t[M * K];
    half_t* h_B = new half_t[N * K];
    half_t* h_C = new half_t[M * N];

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<half_t>((float)(rand() % 100) / 100.0f - 0.5f);
    for (int i = 0; i < N * K; i++) h_B[i] = static_cast<half_t>((float)(rand() % 100) / 100.0f - 0.5f);
    for (int i = 0; i < M * N; i++) h_C[i] = static_cast<half_t>(0.0f);

    half_t *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_B, N * K * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half_t)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * K * sizeof(half_t), cudaMemcpyHostToDevice));

    // Warmup
    gemm_tn(M, N, K, d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    const int WARMUP = 5;
    const int ITERS = 20;

    for (int i = 0; i < WARMUP; i++) {
        gemm_tn(M, N, K, d_A, d_B, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.start();
    for (int i = 0; i < ITERS; i++) {
        gemm_tn(M, N, K, d_A, d_B, d_C);
    }
    timer.stop();
    float avg_ms = timer.elapsed_ms() / ITERS;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (avg_ms * 1e6);

    std::cout << "  平均时间: " << std::fixed << std::setprecision(3) << avg_ms << " ms" << std::endl;
    std::cout << "  性能: " << std::fixed << std::setprecision(1) << gflops << " GFLOPS" << std::endl;

    // 验证
    if (do_verify) {
        CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(half_t), cudaMemcpyDeviceToHost));

        if (M <= 512 && N <= 512 && K <= 512) {
            std::cout << "  CPU 验证中..." << std::endl;
            half_t* h_C_ref = new half_t[M * N];
            cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
            verify(h_C, h_C_ref, M, N);
            delete[] h_C_ref;
        } else {
            std::cout << "  (矩阵过大，跳过 CPU 验证)" << std::endl;
        }
    }
    std::cout << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 12: 性能测试 GEMM" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // 检查 GPU
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name
              << " (SM" << props.major * 10 + props.minor << ")" << std::endl;
    std::cout << std::endl;

    if (props.major < 8) {
        std::cout << "需要 Ampere GPU (SM80+)" << std::endl;
        return 0;
    }

    int M = 1024, N = 1024, K = 1024;
    if (argc >= 2) M = atoi(argv[1]);
    if (argc >= 3) N = atoi(argv[2]);
    if (argc >= 4) K = atoi(argv[3]);

    std::cout << "GEMM: C(" << M << "x" << N << ") = A(" << M << "x" << K << ") * B(" << N << "x" << K << ")" << std::endl;
    std::cout << "Block tile: " << bM << "x" << bN << "x" << bK << std::endl;
    std::cout << "Pipeline depth: " << K_PIPE_MAX << std::endl;
    std::cout << std::endl;

    perf_test(M, N, K);

    // 不同尺寸的测试 (仅在无参数时)
    if (argc < 2) {
        perf_test(256, 256, 256);
        perf_test(512, 512, 512);
        perf_test(2048, 2048, 2048);
        perf_test(4096, 4096, 4096);
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 12 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
