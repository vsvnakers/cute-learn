/**
 * ============================================================================
 * CuTe + MMA 教程 04: 异步拷贝 (cp.async)
 * ============================================================================
 *
 * SM80+ 引入了 cp.async 指令，实现 Global -> Shared 的异步拷贝。
 * 异步拷贝的优势：
 *   1. 不经过寄存器：数据直接从 Global Memory 写入 Shared Memory
 *   2. 计算与传输重叠：可以在等待数据传输的同时执行计算
 *   3. 减少寄存器压力：不需要中间寄存器缓冲
 *
 * 编译：make 04_copy_async
 * 运行：./04_copy_async
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm80.hpp>

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
// 1. cp.async 基本原理
// ============================================================================

void test_cp_async_info() {
    std::cout << "=== 1. cp.async 基本原理 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  cp.async 指令族 (SM80+):" << std::endl;
    std::cout << std::endl;
    std::cout << "  SM80_CP_ASYNC_CACHEALWAYS<TS, TD>:" << std::endl;
    std::cout << "    - 缓存策略: L1 + L2 都缓存" << std::endl;
    std::cout << "    - 适用: 数据会被多次使用" << std::endl;
    std::cout << "    - PTX: cp.async.ca.shared.global" << std::endl;
    std::cout << std::endl;
    std::cout << "  SM80_CP_ASYNC_CACHEGLOBAL<TS, TD>:" << std::endl;
    std::cout << "    - 缓存策略: 仅 L2 缓存" << std::endl;
    std::cout << "    - 适用: 数据只使用一次（流式访问）" << std::endl;
    std::cout << "    - PTX: cp.async.cg.shared.global" << std::endl;
    std::cout << std::endl;
    std::cout << "  SM80_CP_ASYNC_CACHEALWAYS_ZFILL<TS, TD>:" << std::endl;
    std::cout << "    - 缓存策略: L1 + L2，越界填充零" << std::endl;
    std::cout << "    - 适用: 边界处理" << std::endl;
    std::cout << std::endl;
    std::cout << "  支持的数据宽度: 4B, 8B, 16B" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 2. 同步拷贝 vs 异步拷贝
// ============================================================================

void test_sync_vs_async() {
    std::cout << "=== 2. 同步 vs 异步拷贝 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  同步拷贝 (UniversalCopy):" << std::endl;
    std::cout << "    Global -> Register -> Shared" << std::endl;
    std::cout << "    必须等待拷贝完成才能继续" << std::endl;
    std::cout << "    数据经过寄存器中转" << std::endl;
    std::cout << std::endl;

    std::cout << "  异步拷贝 (cp.async):" << std::endl;
    std::cout << "    Global -> Shared (直接)" << std::endl;
    std::cout << "    发起拷贝后可以继续计算" << std::endl;
    std::cout << "    数据不经过寄存器" << std::endl;
    std::cout << std::endl;

    std::cout << "  性能优势:" << std::endl;
    std::cout << "    1. 减少寄存器使用 (不经过寄存器)" << std::endl;
    std::cout << "    2. 计算与传输重叠 (延迟隐藏)" << std::endl;
    std::cout << "    3. 更高的内存带宽利用率" << std::endl;
    std::cout << std::endl;

    std::cout << "  使用条件:" << std::endl;
    std::cout << "    - 需要 SM80+ (Ampere)" << std::endl;
    std::cout << "    - 源地址必须是 Global Memory" << std::endl;
    std::cout << "    - 目标地址必须是 Shared Memory" << std::endl;
    std::cout << "    - 数据大小必须是 4/8/16 bytes" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 3. CuTe 中的异步拷贝 API
// ============================================================================
// CuTe 封装了 cp.async，提供同步机制

void test_cp_async_api() {
    std::cout << "=== 3. CuTe cp.async API ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  CuTe 提供的同步 API:" << std::endl;
    std::cout << std::endl;
    std::cout << "  cute::cp_async_fence():" << std::endl;
    std::cout << "    - 提交当前所有未完成的异步拷贝为一个组" << std::endl;
    std::cout << "    - 对应 PTX: cp.async.commit_group" << std::endl;
    std::cout << std::endl;
    std::cout << "  cute::cp_async_wait<N>():" << std::endl;
    std::cout << "    - 等待直到最多 N 个组未完成" << std::endl;
    std::cout << "    - cp_async_wait<0>(): 等待所有组完成" << std::endl;
    std::cout << "    - cp_async_wait<1>(): 允许 1 个组未完成" << std::endl;
    std::cout << "    - 对应 PTX: cp.async.wait_group N" << std::endl;
    std::cout << std::endl;
    std::cout << "  cp.async.wait_all:" << std::endl;
    std::cout << "    - 等待所有异步拷贝完成" << std::endl;
    std::cout << "    - 对应 PTX: cp.async.wait_all" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 4. 流水线 (Software Pipeline)
// ============================================================================
// 异步拷贝的核心优势是可以实现计算与传输的重叠
// 典型模式：双缓冲 (Double Buffering)

void test_pipeline_info() {
    std::cout << "=== 4. 异步拷贝流水线 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  双缓冲 (Double Buffering) 模式:" << std::endl;
    std::cout << std::endl;
    std::cout << "  __shared__ float smem_buf[2][TILE_SIZE];" << std::endl;
    std::cout << std::endl;
    std::cout << "  // 预取第 0 块" << std::endl;
    std::cout << "  issue_cp_async(smem_buf[0], gmem + 0*TILE);" << std::endl;
    std::cout << "  cp_async_fence();" << std::endl;
    std::cout << std::endl;
    std::cout << "  for (int tile = 0; tile < num_tiles; tile++) {" << std::endl;
    std::cout << "    int cur = tile % 2;" << std::endl;
    std::cout << "    int nxt = (tile + 1) % 2;" << std::endl;
    std::cout << std::endl;
    std::cout << "    // 等待当前缓冲区就绪" << std::endl;
    std::cout << "    cp_async_wait<0>();" << std::endl;
    std::cout << "    __syncthreads();" << std::endl;
    std::cout << std::endl;
    std::cout << "    // 计算 (使用当前缓冲区)" << std::endl;
    std::cout << "    compute(smem_buf[cur]);" << std::endl;
    std::cout << std::endl;
    std::cout << "    // 预取下一块 (异步)" << std::endl;
    std::cout << "    if (tile + 1 < num_tiles)" << std::endl;
    std::cout << "      issue_cp_async(smem_buf[nxt], gmem + (tile+1)*TILE);" << std::endl;
    std::cout << "    cp_async_fence();" << std::endl;
    std::cout << "  }" << std::endl;
    std::cout << std::endl;

    std::cout << "  commit_group / wait_group 机制:" << std::endl;
    std::cout << "    commit_group: 将当前所有未完成的 cp.async 打包为一个组" << std::endl;
    std::cout << "    wait_group<N>: 等待直到最多 N 个组未完成" << std::endl;
    std::cout << std::endl;
    std::cout << "    流水线深度与 wait_group:" << std::endl;
    std::cout << "      深度 2 (双缓冲): wait_group<1>" << std::endl;
    std::cout << "      深度 3 (三缓冲): wait_group<2>" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 5. 异步拷贝在 GEMM 中的应用
// ============================================================================

void test_gemm_pipeline() {
    std::cout << "=== 5. GEMM 中的异步拷贝 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  典型 GEMM 的数据流:" << std::endl;
    std::cout << std::endl;
    std::cout << "  Global Memory (A, B)" << std::endl;
    std::cout << "       |" << std::endl;
    std::cout << "       | cp.async (异步)" << std::endl;
    std::cout << "       v" << std::endl;
    std::cout << "  Shared Memory (smem_A, smem_B)" << std::endl;
    std::cout << "       |" << std::endl;
    std::cout << "       | ldmatrix / ld.global (同步)" << std::endl;
    std::cout << "       v" << std::endl;
    std::cout << "  Registers (frag_A, frag_B)" << std::endl;
    std::cout << "       |" << std::endl;
    std::cout << "       | mma.sync (Tensor Core)" << std::endl;
    std::cout << "       v" << std::endl;
    std::cout << "  Registers (frag_C)" << std::endl;
    std::cout << "       |" << std::endl;
    std::cout << "       | st.global (同步)" << std::endl;
    std::cout << "       v" << std::endl;
    std::cout << "  Global Memory (C)" << std::endl;
    std::cout << std::endl;

    std::cout << "  优化要点:" << std::endl;
    std::cout << "    1. 使用双缓冲重叠 cp.async 和 MMA" << std::endl;
    std::cout << "    2. cp.async 不经过寄存器，减少寄存器压力" << std::endl;
    std::cout << "    3. 使用 swizzle 避免 shared memory bank conflict" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 6. 异步拷贝的实际使用示例
// ============================================================================
// 在 CuTe 中，异步拷贝通过 Copy_Atom 和 TiledCopy 使用
// 与同步拷贝的唯一区别是 CopyOp 类型不同

__global__ void async_copy_demo(
    const float* __restrict__ gmem_src,
    float* __restrict__ smem_dst)
{
    // Shared memory
    __shared__ float smem[1024];

    // 使用 AutoVectorizingCopy 作为 CopyOp
    // CuTe 会自动选择最优的拷贝宽度 (4/8/16 bytes)
    // 对于 cp.async (Global → Shared 不经过寄存器):
    //   使用 SM80_CP_ASYNC_CACHEALWAYS<uint4, uint4>
    //   但 cp.async 只支持 Global → Shared 方向
    //   本示例使用同步拷贝演示基本流程
    using CopyOp = AutoVectorizingCopy;
    using Atom = Copy_Atom<CopyOp, float>;

    auto tiled_copy = make_tiled_copy(
        Atom{},
        make_layout(make_shape(Int<256>{})),
        make_layout(make_shape(Int<4>{}))
    );

    // 创建 Tensor
    // 注意: Global 和 Shared 的 Tensor 使用不同的指针类型
    auto g_tensor = make_tensor(make_gmem_ptr(gmem_src),
                                make_layout(make_shape(Int<1024>{})));
    auto s_tensor = make_tensor(make_smem_ptr(smem),
                                make_layout(make_shape(Int<1024>{})));

    // 获取线程视图
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto thr_g = thr_copy.partition_S(g_tensor);
    auto thr_s = thr_copy.partition_D(s_tensor);

    // 创建寄存器 fragment
    auto frag = make_fragment_like(thr_g);

    // 执行拷贝: Global -> Shared
    copy(tiled_copy, thr_g, thr_s);

    // 确保后续读取能看到最新数据
    __syncthreads();

    // 从 Shared Memory 读取并写回 Global Memory
    auto s_read = make_tensor(make_smem_ptr(smem),
                              make_layout(make_shape(Int<1024>{})));
    auto g_write = make_tensor(make_gmem_ptr(smem_dst),
                               make_layout(make_shape(Int<1024>{})));

    auto thr_s_read = thr_copy.partition_S(s_read);
    auto thr_g_write = thr_copy.partition_D(g_write);
    auto frag2 = make_fragment_like(thr_s_read);

    // Shared -> Register
    copy(tiled_copy, thr_s_read, frag2);
    // Register -> Global
    copy(tiled_copy, frag2, thr_g_write);
}

void test_async_copy_demo() {
    std::cout << "=== 6. 异步拷贝实际示例 ===" << std::endl;
    std::cout << std::endl;

    const int N = 1024;
    float h_src[1024], h_dst[1024];
    for (int i = 0; i < N; i++) h_src[i] = (float)i;

    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));

    async_copy_demo<<<1, 256>>>(d_src, d_dst);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_dst[i] - h_src[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << "  异步拷贝结果: " << (correct ? "正确" : "错误") << std::endl;
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 04: 异步拷贝 (cp.async)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_cp_async_info();
    test_sync_vs_async();
    test_cp_async_api();
    test_pipeline_info();
    test_gemm_pipeline();
    test_async_copy_demo();

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 04 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
