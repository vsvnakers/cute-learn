/**
 * CuTe Elementwise Add Kernel — 全显式版 + 多 N 带宽测试
 *
 * 所有 CuTe 的隐式默认参数都显式写出来了。
 * 运行: ./elementwise_add        (默认跑一组 N 值)
 *       ./elementwise_add 1048576 (只跑指定 N)
 *
 * 详细原理见: cute_elementwise_tutorial.md
 */

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace cute;

// ============================================================================
// GPU Timer
// ============================================================================
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start(cudaStream_t s = 0) { cudaEventRecord(start_, s); }
    void stop(cudaStream_t s = 0)  { cudaEventRecord(stop_, s); }
    float elapsed_ms() {
        cudaEventSynchronize(stop_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// CuTe Elementwise Add Kernel
// ============================================================================
//
// 每个 block 处理 1 个 tile (1024 个 float)。
// 256 线程 × 4 float/thread = 1024 float/tile = 4096 bytes/tile。
//
template <class TiledCopy,
          class TensorTiledA, class TensorTiledB, class TensorTiledC>
__global__ void elementwise_add_kernel(
    TiledCopy    tiled_copy,      // 分块拷贝计划
    TensorTiledA tiled_A,         // (TILE_SIZE, num_tiles) — A 的分块视图
    TensorTiledB tiled_B,         // (TILE_SIZE, num_tiles) — B 的分块视图
    TensorTiledC tiled_C,         // (TILE_SIZE, num_tiles) — C 的分块视图
    int          num_tiles)       // tile 总数
{
    int tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) return;

    // ── Step 1: 获取当前线程的"拷贝切片" ──────────────────────────────
    // tiled_copy 内部记录了 (ThreadLayout, ValueLayout, CopyAtom) 的完整信息。
    // get_thread_slice(threadIdx.x) 返回一个 ThrCopy 对象，
    // 它知道"第 threadIdx.x 个线程应该搬运 tile 中的哪些元素"。
    //
    // 用 auto 让编译器推导类型，实际类型是:
    //   typename TiledCopy::ThrCopy (一个记录了线程-数据映射的视图对象)
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

    // ── Step 2: 从 2D tiled 张量中取出当前 tile ──────────────────────
    // tiled_A 的形状是 (TILE_SIZE, num_tiles)，即 (1024, num_tiles)。
    // 使用 CuTe 的"坐标切片"语法: (_, tile_idx)
    //   _        → 保留第 0 维（1024 个元素全保留）
    //   tile_idx → 选择第 1 维的第 tile_idx 个 tile
    //
    // 结果 tile_A 的形状是 (1024,)，是 tiled_A 的一个"列切片"。
    // 注意: tile_A 不拷贝数据，它只是 tiled_A 的一个视图（view），
    //       底层指针指向 tiled_A 中第 tile_idx 个 tile 的起始位置。
    auto tile_A = tiled_A(_, tile_idx);   // (TILE_SIZE,)
    auto tile_B = tiled_B(_, tile_idx);   // (TILE_SIZE,)
    auto tile_C = tiled_C(_, tile_idx);   // (TILE_SIZE,)

    // ── Step 3: 把 tile 分配给当前线程 ──────────────────────────────
    // partition_S / partition_D 根据 thr_copy 的信息，把 tile 切成当前线程负责的部分。
    //
    // partition_S: 切源张量（读取用）
    // partition_D: 切目标张量（写入用）
    //
    // 对于我们的配置 (256 threads, 4 values/thread):
    //   thr_A 的形状是 (4,) — 当前线程负责 tile 中连续的 4 个 float
    //   线程 0 负责 [0,1,2,3]，线程 1 负责 [4,5,6,7]，...
    //
    // 注意: "partition" 不拷贝数据，它创建一个"视图"，
    //       告诉 copy() 函数"这个线程应该从哪里读/往哪里写"。
    auto thr_A = thr_copy.partition_S(tile_A);   // (4,)
    auto thr_B = thr_copy.partition_S(tile_B);   // (4,)
    auto thr_C = thr_copy.partition_D(tile_C);   // (4,)

    // ── Step 4: 创建寄存器张量（fragment）──────────────────────────
    // make_fragment_like(thr_A) 创建一个形状和 thr_A 相同的张量，
    // 但数据存储在 GPU 寄存器中（而非全局内存）。
    //
    // 寄存器是 GPU 上最快的存储:
    //   寄存器延迟: ~4 cycle
    //   全局内存延迟: ~400 cycle
    //   差距: 100 倍
    //
    // 我们需要 3 个 fragment:
    //   frag_A: 存放从 A 加载的数据
    //   frag_B: 存放从 B 加载的数据
    //   frag_C: 存放计算结果 A+B
    auto frag_A = make_fragment_like(thr_A);   // (4,) in registers
    auto frag_B = make_fragment_like(thr_B);   // (4,) in registers
    auto frag_C = make_fragment_like(thr_C);   // (4,) in registers

    // ── Step 5: 向量化加载 ─────────────────────────────────────────
    // copy(tiled_copy, src, dst) 执行拷贝。
    //
    // tiled_copy 内部的 CopyAtom 是 UniversalCopy<uint4>:
    //   uint4 = 4 个 unsigned int = 128 bit = 16 bytes
    //   对 float 类型: 16 bytes / 4 bytes = 4 个 float 一次加载
    //
    // 编译器生成的 PTX 指令:
    //   LDG.E.128  — 128-bit 全局内存加载（对应一条 CUDA 指令）
    //   而不是 LDG.E.32（32-bit，只能加载 1 个 float）
    //
    // 执行过程:
    //   1. copy 读取 thr_A 的"地址描述"（它知道当前线程的数据在 GMEM 的哪里）
    //   2. copy 读取 frag_A 的"地址描述"（它知道寄存器在哪里）
    //   3. 生成一条 LDG.128 指令，从 GMEM 搬运 16 bytes 到寄存器
    copy(tiled_copy, thr_A, frag_A);   // GMEM → Register: 加载 A[i..i+3]
    copy(tiled_copy, thr_B, frag_B);   // GMEM → Register: 加载 B[i..i+3]

    // ── Step 6: 寄存器计算 ─────────────────────────────────────────
    // CUTE_UNROLL = #pragma unroll，告诉编译器展开循环。
    // 展开后变成 4 条独立的 FADD 指令，可以流水线并行。
    //
    // size(frag_C) 返回 fragment 中的元素总数（这里是 4）。
    // frag_C(i) 访问 fragment 中的第 i 个元素（在寄存器中，零延迟）。
    CUTE_UNROLL
    for (int i = 0; i < size(frag_C); ++i) {
        frag_C(i) = frag_A(i) + frag_B(i);
    }

    // ── Step 7: 向量化写回 ─────────────────────────────────────────
    // 和 Step 5 对称，使用 STG.128 指令一次性写回 4 个 float 到 GMEM。
    copy(tiled_copy, frag_C, thr_C);   // Register → GMEM: 写回 C[i..i+3]
}

// ============================================================================
// 朴素参考 kernel
// ============================================================================
__global__ void naive_add_kernel(float const* A, float const* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// ============================================================================
// 带宽测试函数
// ============================================================================
void benchmark_N(int N, GpuTimer& timer)
{
    // ── 1. 分配内存 ─────────────────────────────────────────────────
    size_t bytes = (size_t)N * sizeof(float);
    float *h_A     = (float*)malloc(bytes);
    float *h_B     = (float*)malloc(bytes);
    float *h_C     = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // ── 2. 初始化 ───────────────────────────────────────────────────
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ── 3. CuTe 参数（全显式）──────────────────────────────────────
    constexpr int TILE_SIZE   = 1024;  // 每个 tile 的 float 数量
    constexpr int NUM_THREADS = 256;   // 每个 block 的线程数
    constexpr int VEC_WIDTH   = 4;     // 每个线程处理的 float 数量

    // 3a. Layout — 全显式写出 Shape 和 Stride
    //
    // make_layout(make_shape(X), make_stride(Y)):
    //   Shape  = X: 每个维度有多少个元素
    //   Stride = Y: 维度移动 1，物理地址变化多少
    //
    // 之前写的 make_layout(Int<256>{}) 是简写，完整形式如下:

    // 线程布局: 256 个线程排成 1D
    //   Shape  = (256,)  — 256 个线程
    //   Stride = (1,)    — 线程 ID 连续: 线程 0→偏移0, 线程 1→偏移1, ...
    auto thr_layout = make_layout(
        make_shape(Int<NUM_THREADS>{}),    // Shape:  (256,)
        make_stride(Int<1>{})              // Stride: (1,)  — 连续排列
    );
    // thr_layout 将坐标映射为: coord(0)→0, coord(1)→1, ..., coord(255)→255

    // 值布局: 每个线程处理 4 个连续的 float
    //   Shape  = (4,) — 4 个值
    //   Stride = (1,) — 连续: 值 0→偏移0, 值 1→偏移1, 值 2→偏移2, 值 3→偏移3
    auto val_layout = make_layout(
        make_shape(Int<VEC_WIDTH>{}),      // Shape:  (4,)
        make_stride(Int<1>{})              // Stride: (1,)  — 连续排列
    );
    // val_layout 将坐标映射为: coord(0)→0, coord(1)→1, coord(2)→2, coord(3)→3

    // 3b. Copy_Atom — 显式写出操作类型和数据类型
    //
    // UniversalCopy<AccessType>:
    //   AccessType = uint4 = 4 × unsigned int = 128 bit = 16 bytes
    //   这告诉 CuTe "每次内存操作搬运 16 字节"
    //
    // 第二个模板参数 float:
    //   告诉 CuTe "这些字节应该被解释为 float"
    //   16 bytes / sizeof(float) = 16 / 4 = 4 个 float
    //
    // 组合效果: 一次 copy 操作 = 1 条 LDG.128 指令 = 搬运 4 个 float
    using CopyOp  = UniversalCopy<uint4>;   // 128-bit 访问宽度
    using Element = float;                  // 数据类型: float (4 bytes)
    using Atom    = Copy_Atom<CopyOp, Element>;
    // Copy_Atom<UniversalCopy<uint4>, float> 的含义:
    //   "用 128-bit 的宽度搬运 float 类型数据"
    //   "每次搬运 128/32 = 4 个 float"

    // 3c. TiledCopy — 把 Copy_Atom 铺满整个 tile
    //
    // make_tiled_copy(atom, thread_layout, value_layout):
    //   atom:          单个线程的拷贝策略（128-bit float4）
    //   thr_layout:    256 个线程怎么排列
    //   val_layout:    每个线程处理几个值
    //
    // 生成的 TiledCopy 描述了:
    //   "256 个线程，每个用 1 条 LDG.128 搬运 4 个 float"
    //   "总共搬运 256 × 4 = 1024 个 float = 4096 bytes = 4KB"
    auto tiled_copy = make_tiled_copy(
        Atom{},          // Copy_Atom: 每线程 128-bit float4 拷贝
        thr_layout,      // 线程布局: (256,) stride=(1,)
        val_layout       // 值布局:   (4,)   stride=(1,)
    );
    // tiled_copy 内部存储了:
    //   - ThreadLayout: 256 个线程的排列方式
    //   - ValueLayout: 每个线程处理 4 个值
    //   - CopyAtom: 用 128-bit 指令搬运 float
    //   - 它能计算出: 第 i 个线程应该访问 tile 中的哪些地址

    // 3d. Tiled 张量 — 全显式写出 tiled_divide 的参数
    //
    // 第一步: make_gmem_ptr — 把原始指针包装成 CuTe 的全局内存指针
    //   make_gmem_ptr(d_A) 把 float* 包装成 gmem_ptr<float>
    //   这告诉 CuTe "这个指针指向 GPU 全局内存"
    //   gmem_ptr 是一个迭代器类型，CuTe 通过它来生成 LDG/STG 指令
    //
    // 第二步: make_layout — 创建 1D 布局
    //   make_layout(make_shape(N), make_stride(Int<1>{}))
    //     Shape  = N（动态大小，运行时确定）
    //     Stride = 1（静态，编译时确定）— 连续存储
    //   等价于 make_layout(N)，但这里显式写出 stride
    //
    // 第三步: make_tensor — 创建张量 = 指针 + 布局
    //   tensor(i) = *(gmem_ptr + i * stride) = *(d_A + i * 1) = d_A[i]
    //
    // 第四步: tiled_divide — 把 1D 张量折叠成 2D
    //   输入: (N,) 的 1D 张量
    //   输出: (TILE_SIZE, num_tiles) 的 2D 张量
    //   内部: 创建 ComposedLayout，坐标 (i,j) → 物理地址 j*TILE_SIZE + i
    //   等价于把原始数组每 1024 个元素切成一个 tile
    //
    // 注意: gmem_ptr 和 Tensor 的类型太复杂（嵌套模板），
    //       无法手写，必须用 auto 让编译器推导。
    //       但 make_shape / make_stride 的参数我们已经显式写出了。

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // 1D 张量: (N,) stride=(1,)
    auto tensor_A = make_tensor(
        make_gmem_ptr(d_A),                    // gmem_ptr<float>: 指向 GPU 全局内存
        make_layout(
            make_shape(N),                     // Shape:  N 个元素（动态）
            make_stride(Int<1>{})              // Stride: 1（连续存储）
        )
    );
    // tensor_A(i) = d_A[i]

    auto tensor_B = make_tensor(
        make_gmem_ptr(d_B),
        make_layout(make_shape(N), make_stride(Int<1>{}))
    );

    auto tensor_C = make_tensor(
        make_gmem_ptr(d_C),
        make_layout(make_shape(N), make_stride(Int<1>{}))
    );

    // tiled_divide: (N,) → (1024, num_tiles)
    //   tile_shape = make_shape(Int<1024>{}) = 1024（静态 tile 大小）
    //   坐标 (i, j) → d_A[j * 1024 + i]
    auto tiled_A = tiled_divide(
        tensor_A,                            // 输入: (N,) 1D 张量
        make_shape(Int<TILE_SIZE>{})         // tile 大小: 1024（静态）
    );
    // tiled_A 形状: (1024, num_tiles)
    // tiled_A(i, j) = d_A[j * 1024 + i]

    auto tiled_B = tiled_divide(
        tensor_B,
        make_shape(Int<TILE_SIZE>{})
    );

    auto tiled_C = tiled_divide(
        tensor_C,
        make_shape(Int<TILE_SIZE>{})
    );

    // ── 4. 正确性验证 ───────────────────────────────────────────────
    // 朴素 kernel
    {
        dim3 block(NUM_THREADS);
        dim3 grid((N + NUM_THREADS - 1) / NUM_THREADS);
        naive_add_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C_ref, d_C, bytes, cudaMemcpyDeviceToHost));
    }
    // CuTe kernel
    {
        dim3 block(NUM_THREADS);
        dim3 grid(num_tiles);
        elementwise_add_kernel<<<grid, block>>>(
            tiled_copy, tiled_A, tiled_B, tiled_C, num_tiles);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    }
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = fabsf(h_C[i] - h_C_ref[i]);
        if (err > max_err) max_err = err;
    }
    if (max_err >= 1e-4f) {
        printf("  [ERROR] N=%d max_err=%e\n", N, max_err);
    }

    // ── 5. 性能测试 ─────────────────────────────────────────────────
    const int warmup = 10;
    const int iterations = 100;
    double total_bytes = (double)N * 12.0;  // 2R + 1W

    dim3 block(NUM_THREADS);
    dim3 grid(num_tiles);

    // 预热
    for (int i = 0; i < warmup; ++i)
        elementwise_add_kernel<<<grid, block>>>(
            tiled_copy, tiled_A, tiled_B, tiled_C, num_tiles);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    timer.start();
    for (int i = 0; i < iterations; ++i)
        elementwise_add_kernel<<<grid, block>>>(
            tiled_copy, tiled_A, tiled_B, tiled_C, num_tiles);
    timer.stop();
    float ms = timer.elapsed_ms() / iterations;
    double bw = total_bytes / (ms * 1e-3) / 1e9;

    // 打印结果
    double mb = N * 4.0 / (1024 * 1024);
    printf("  N = %10d (%7.1f MB)  |  %8.4f ms  |  %7.1f GB/s  (%4.1f%%)\n",
           N, mb, ms, bw, bw / 336.0 * 100.0);

    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv)
{
    printf("================================================================\n");
    printf("  CuTe Elementwise Add — Multi-N Bandwidth Benchmark\n");
    printf("  RTX 3060 Laptop peak: ~336 GB/s\n");
    printf("================================================================\n\n");

    GpuTimer timer;

    if (argc >= 2) {
        // 单个 N 模式
        int N = atoi(argv[1]);
        printf("  %-12s  %-11s  %-10s  %-14s  %s\n",
               "N", "Per-tensor", "Time", "Bandwidth", "Peak%");
        printf("  %-12s  %-11s  %-10s  %-14s  %s\n",
               "------", "---------", "------", "---------", "------");
        benchmark_N(N, timer);
    } else {
        // 多 N 扫描模式
        // 从 4KB 到 1GB，覆盖各种规模
        std::vector<int> Ns = {
            1024,             //    4 KB — 1 个 tile
            4096,             //   16 KB — 4 个 tile
            16384,            //   64 KB
            65536,            //  256 KB
            262144,           //    1 MB
            1048576,          //    4 MB
            4194304,          //   16 MB
            16777216,         //   64 MB
            67108864,         //  256 MB
            134217728,        //  512 MB
            268435456,        //    1 GB
        };

        printf("  %-12s  %-11s  %-10s  %-14s  %s\n",
               "N", "Per-tensor", "Time", "Bandwidth", "Peak%");
        printf("  %-12s  %-11s  %-10s  %-14s  %s\n",
               "------", "---------", "------", "---------", "------");

        for (int N : Ns) {
            benchmark_N(N, timer);
        }

        printf("\n  Notes:\n");
        printf("  - Small N: kernel launch overhead dominates, low bandwidth\n");
        printf("  - Medium N: L2 cache helps, bandwidth increases\n");
        printf("  - Large N: memory-bound, bandwidth plateaus near peak\n");
        printf("  - 70%% target = %.1f GB/s\n", 336.0 * 0.7);
    }

    return 0;
}
