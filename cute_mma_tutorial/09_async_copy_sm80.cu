/**
 * ============================================================================
 * CuTe + MMA 教程 09: SM80 cp.async 异步拷贝实战
 * ============================================================================
 *
 * 本教程展示 SM80+ cp.async 指令的实际使用:
 *   1. SM80_CP_ASYNC_CACHEALWAYS CopyAtom 创建
 *   2. make_tiled_copy 创建 TiledCopy (G2S 方向)
 *   3. Global→Shared 异步拷贝 kernel (数据验证)
 *   4. cp_async_fence / cp_async_wait<0> 同步机制
 *   5. 对比同步拷贝与异步拷贝的代码差异
 *
 * 关键概念:
 *   - cp.async: Global→Shared 直接传输，不经过寄存器
 *   - cp_async_fence(): 提交当前异步拷贝为一个组
 *   - cp_async_wait<N>(): 等待直到最多 N 个组完成
 *   - cp.async 只支持 Global→Shared 方向
 *
 * 编译: make 09_async_copy_sm80
 * 运行: ./09_async_copy_sm80
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
// 配置
// ============================================================================

// make_tiled_copy 的三个参数决定tile大小:
//   atom:           Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>
//                   → 每条cp.async指令搬16B = 8个half
//   thread_layout:  Layout<Shape<_16,_8>, Stride<_8,_1>>
//                   → 128线程排成16行8列, 行内stride=1(连续线程)
//   value_layout:   Layout<Shape<_1,_8>>
//                   → 每线程8个half (1x8)
// 总tile: 16x64 = 1024 half = 128线程 * 8值/线程

constexpr int TILE_M = 16;
constexpr int TILE_N = 64;
constexpr int BLOCK_SIZE = 128;

// ============================================================================
// 1. cp.async 异步拷贝 Kernel
// ============================================================================
//
// 数据流:
//   Global → Shared (cp.async, 不经过寄存器)
//   Shared → Global (同步, 手动拷贝, 验证用)
//
// 注意: cp.async 只支持 Global→Shared 方向!
//       Shared→Global 必须使用同步拷贝

__global__ void async_copy_kernel(
    const half_t* __restrict__ src,  // 源Global内存
    half_t* __restrict__ dst,        // 目标Global内存 (验证用)
    int total_elems)                 // 源/目标总元素数
{
    // Shared Memory (__align__(16) 是 cp.async 的硬件要求)
    extern __shared__ __align__(16) half_t smem[];

    // ---- G2S CopyAtom: cp.async 128-bit ----
    // SM80_CP_ASYNC_CACHEALWAYS<uint128_t>: 每次传输 16 bytes
    // 对于 half_t: 一次传输 8 个元素
    using G2SAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>;

    // make_tiled_copy参数: (atom, thread_layout, value_layout)
    // thread_layout: 128线程排成16行8列, K-major步长
    // value_layout:  每线程8个half (128-bit/16B per thread)
    auto g2s_copy = make_tiled_copy(
        G2SAtom{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}
    );

    constexpr int TILE_ELEMS = TILE_M * TILE_N;

    int offset = blockIdx.x * TILE_ELEMS;
    if (offset >= total_elems) return;

    // ---- 2D Tensor ----
    auto g_src = make_tensor(make_gmem_ptr(src + offset),
                             make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                             make_stride(Int<TILE_N>{}, Int<1>{}));
    auto s_dst = make_tensor(make_smem_ptr(smem),
                             make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                             make_stride(Int<TILE_N>{}, Int<1>{}));

    // ---- 线程视图 ----
    auto thr_copy = g2s_copy.get_slice(threadIdx.x);
    auto tSrc = thr_copy.partition_S(g_src);
    auto tDst = thr_copy.partition_D(s_dst);

    // ---- 异步拷贝: Global → Shared ----
    // copy() 内部使用 cp.async 指令
    // 数据直接从 Global Memory 写入 Shared Memory, 不经过寄存器
    cute::copy(g2s_copy, tSrc, tDst);

    // ---- 同步 ----
    // cp_async_fence(): 提交当前异步拷贝为一个组 (commit_group)
    // cp_async_wait<0>(): 等待所有组完成
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // ---- 读回: Shared → Global (手动拷贝, 验证用) ----
    // cp.async 只支持 G→S 方向, S→G 使用普通拷贝
    for (int i = threadIdx.x; i < TILE_ELEMS; i += BLOCK_SIZE) {
        dst[offset + i] = smem[i];
    }
}

// ============================================================================
// 2. 同步拷贝 Kernel (对比)
// ============================================================================
//
// 使用 UniversalCopy (同步拷贝):
//   Global → Register → Shared
//   必须等待拷贝完成才能继续

__global__ void sync_copy_kernel(
    const half_t* __restrict__ src,  // 源Global内存
    half_t* __restrict__ dst,        // 目标Global内存 (验证用)
    int total_elems)                 // 总元素数
{
    extern __shared__ half_t smem[];  // 同步拷贝不需要__align__(16)

    // UniversalCopy: 同步拷贝, 数据路径 Global→Register→Shared
    using CopyAtom = Copy_Atom<UniversalCopy<uint128_t>, half_t>;

    auto copy_op = make_tiled_copy(
        CopyAtom{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}
    );

    constexpr int TILE_ELEMS = TILE_M * TILE_N;

    int offset = blockIdx.x * TILE_ELEMS;
    if (offset >= total_elems) return;

    auto g_src = make_tensor(make_gmem_ptr(src + offset),
                             make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                             make_stride(Int<TILE_N>{}, Int<1>{}));
    auto s_dst = make_tensor(make_smem_ptr(smem),
                             make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                             make_stride(Int<TILE_N>{}, Int<1>{}));

    auto thr_copy = copy_op.get_slice(threadIdx.x);
    auto tSrc = thr_copy.partition_S(g_src);
    auto tDst = thr_copy.partition_D(s_dst);

    // 同步拷贝: Global → Register → Shared
    cute::copy(copy_op, tSrc, tDst);
    __syncthreads();

    // 读回
    for (int i = threadIdx.x; i < TILE_ELEMS; i += BLOCK_SIZE) {
        dst[offset + i] = smem[i];
    }
}

// ============================================================================
// 3. 验证
// ============================================================================

bool verify_copy(const half_t* src, const half_t* dst, int N) {
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float s = static_cast<float>(src[i]);
        float d = static_cast<float>(dst[i]);
        if (fabsf(s - d) > 1e-5) {
            if (errors < 5) {
                std::cerr << "  Error at [" << i << "]: src=" << s << " dst=" << d << std::endl;
            }
            errors++;
        }
    }
    if (errors > 0) {
        std::cerr << "  Total errors: " << errors << " / " << N << std::endl;
    }
    return errors == 0;
}

// ============================================================================
// 4. 测试
// ============================================================================

void test_copy(bool async) {
    const char* name = async ? "cp.async 异步拷贝" : "同步拷贝";
    std::cout << "=== 测试: " << name << " ===" << std::endl;

    constexpr int TILE_ELEMS = TILE_M * TILE_N;   // 1024
    const int N = TILE_ELEMS * 1024;               // 1024个tile = 1M元素

    half_t* h_src = new half_t[N];
    half_t* h_dst = new half_t[N];
    for (int i = 0; i < N; i++) {
        h_src[i] = static_cast<half_t>((float)(i % 1000) / 1000.0f);
        h_dst[i] = static_cast<half_t>(0.0f);
    }

    half_t *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(half_t)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(half_t), cudaMemcpyHostToDevice));

    int grid_size = N / TILE_ELEMS;                // 1024个block
    int smem_bytes = TILE_ELEMS * sizeof(half_t);

    // cp.async要求shared memory 16字节对齐
    if (async) smem_bytes = (smem_bytes + 15) & ~15;

    // kernel启动: <<<grid, block, shared_mem_bytes>>>
    if (async) {
        async_copy_kernel<<<grid_size, BLOCK_SIZE, smem_bytes>>>(d_src, d_dst, N);
    } else {
        sync_copy_kernel<<<grid_size, BLOCK_SIZE, smem_bytes>>>(d_src, d_dst, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(half_t), cudaMemcpyDeviceToHost));

    bool correct = verify_copy(h_src, h_dst, N);
    std::cout << "  结果: " << (correct ? "验证通过!" : "验证失败!") << std::endl;
    std::cout << std::endl;

    delete[] h_src;
    delete[] h_dst;
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

// ============================================================================
// 5. API 说明
// ============================================================================

void print_api_info() {
    std::cout << "=== cp.async API 说明 ===" << std::endl;
    std::cout << std::endl;

    std::cout << "  CopyAtom:" << std::endl;
    std::cout << "    SM80_CP_ASYNC_CACHEALWAYS<uint128_t, half_t>" << std::endl;
    std::cout << "    - 128-bit (16 bytes) 每次传输 = 8 个 half" << std::endl;
    std::cout << "    - CACHEALWAYS: L1+L2 缓存" << std::endl;
    std::cout << std::endl;

    std::cout << "  TiledCopy:" << std::endl;
    std::cout << "    Thread layout: 16x8 (128 threads)" << std::endl;
    std::cout << "    Value layout: 1x8 (8 half per thread)" << std::endl;
    std::cout << "    Tile: 16x64 = 1024 elements" << std::endl;
    std::cout << std::endl;

    std::cout << "  同步 API:" << std::endl;
    std::cout << "    cp_async_fence():     提交异步拷贝组" << std::endl;
    std::cout << "    cp_async_wait<0>():   等待所有组完成" << std::endl;
    std::cout << "    cp_async_wait<N>():   允许 N 个组未完成" << std::endl;
    std::cout << std::endl;

    std::cout << "  关键区别:" << std::endl;
    std::cout << "    同步 (UniversalCopy): Global → Register → Shared" << std::endl;
    std::cout << "    异步 (cp.async):      Global → Shared (直接)" << std::endl;
    std::cout << "    异步优势: 不经过寄存器, 可与计算重叠" << std::endl;
    std::cout << std::endl;

    std::cout << "  注意:" << std::endl;
    std::cout << "    cp.async 只支持 Global → Shared 方向!" << std::endl;
    std::cout << "    Shared → Global 必须使用同步拷贝" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 09: cp.async 异步拷贝" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    print_api_info();
    test_copy(true);   // 异步
    test_copy(false);  // 同步

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 09 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
