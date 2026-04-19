// PTX 进阶代码示例

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// ============================================================================
// 示例 1: 使用 PTX 实现共享内存优化
// ============================================================================

/**
 * 使用共享内存的向量加法
 * 演示 shared 内存空间的 PTX 操作
 */
__global__ void ptx_shared_memory_demo(float* d_out, const float* d_in, int n) {
    __shared__ float shared_data[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        // 从全局内存加载到共享内存
        float val = d_in[idx];

        // 使用 PTX 存储到共享内存
        asm volatile("st.shared.f32 [%0], %1;"
                     : : "r"((unsigned int)(tid * sizeof(float))), "f"(val));

        __syncthreads();

        // 从共享内存加载
        float shared_val;
        asm volatile("ld.shared.f32 %0, [%1];"
                     : "=f"(shared_val) : "r"((unsigned int)(tid * sizeof(float))));

        d_out[idx] = shared_val * 2.0f;
    }
}

// ============================================================================
// 示例 2: 使用 PTX 实现原子操作
// ============================================================================

/**
 * 使用 PTX 实现原子加法
 * 对应的 PTX 指令：atom.add.u32 %r, [%rd], %r2;
 */
__device__ unsigned int ptx_atomic_add(unsigned int* addr, unsigned int val) {
    unsigned int ret;
    asm volatile("atom.global.add.u32 %0, [%1], %2;"
                 : "=r"(ret)
                 : "l"(addr), "r"(val)
                 : "memory");
    return ret;
}

/**
 * 使用 PTX 实现原子 CAS (Compare-And-Swap)
 * 对应的 PTX 指令：atom.cas.u32 %r, [%rd], %r2, %r3;
 */
__device__ unsigned int ptx_atomic_cas(
    unsigned int* addr,
    unsigned int compare,
    unsigned int swap
) {
    unsigned int ret;
    asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;"
                 : "=r"(ret)
                 : "l"(addr), "r"(compare), "r"(swap)
                 : "memory");
    return ret;
}

/**
 * 使用 PTX 实现原子最小值
 * 对应的 PTX 指令：atom.min.u32 %r, [%rd], %r2;
 */
__device__ unsigned int ptx_atomic_min(unsigned int* addr, unsigned int val) {
    unsigned int ret;
    asm volatile("atom.global.min.u32 %0, [%1], %2;"
                 : "=r"(ret)
                 : "l"(addr), "r"(val)
                 : "memory");
    return ret;
}

__global__ void ptx_atomic_demo(unsigned int* d_counter, unsigned int* d_result) {
    int tid = threadIdx.x;

    // 原子加法演示
    unsigned int old_val = ptx_atomic_add(d_counter, 1);

    if (tid == 0) {
        printf("原子加法：旧值=%u, 新值=%u\n", old_val, old_val + 1);
    }

    // 原子 CAS 演示
    unsigned int expected = 0;
    unsigned int new_val = 100;
    unsigned int cas_result = ptx_atomic_cas(d_counter, expected, new_val);

    if (tid == 0) {
        printf("原子 CAS: 期望=%u, 实际=%u, 成功=%s\n",
               expected, cas_result, cas_result == expected ? "是" : "否");
    }
}

// ============================================================================
// 示例 3: 使用 PTX 实现屏障同步
// ============================================================================

/**
 * 使用 PTX 屏障同步
 * 对应的 PTX 指令：bar.sync 0;
 */
__global__ void ptx_barrier_demo() {
    __shared__ int shared_data[32];
    __shared__ unsigned int bar[1];  // Named barrier storage

    int tid = threadIdx.x;

    // 阶段 1: 写入数据
    shared_data[tid] = tid * 2;

    // 使用 PTX 屏障同步 (barrier ID 0, thread count = blockDim.x)
    asm volatile("bar.sync 0, %0;" : : "r"(blockDim.x));

    // 阶段 2: 读取其他线程的数据
    int sum = 0;
    for (int i = 0; i < blockDim.x; i++) {
        sum += shared_data[i];
    }

    if (tid == 0) {
        printf("屏障同步演示：sum(0..%d)*2 = %d\n", blockDim.x - 1, sum);
    }
}

// ============================================================================
// 示例 4: 使用 PTX 实现内存屏障
// ============================================================================

/**
 * 使用 PTX 内存屏障
 * bar.sync: 块内屏障
 * membar.cta: CTA 内存屏障
 * membar.gl: 全局内存屏障
 */
__global__ void ptx_membar_demo(int* d_data, int* d_flag) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // 写入数据
        d_data[0] = 42;
        d_data[1] = 100;

        // 内存屏障：确保前面的写入完成
        asm volatile("membar.gl;");

        // 设置标志
        *d_flag = 1;

        printf("内存屏障演示：数据已写入，标志已设置\n");
    }
}

// ============================================================================
// 示例 5: 使用 PTX 实现向量加载/存储
// ============================================================================

/**
 * 使用 PTX 向量加载 (128 位)
 * 对应的 PTX 指令：ld.global.v4.f32 {%v4f}, [%rd];
 */
__global__ void ptx_vector_load_demo(float* d_out, const float* d_in, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n) {
        // 使用向量加载 (一次加载 4 个 float)
        float4 val;
        asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                     : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                     : "l"(d_in + idx));

        // 处理数据
        val.x *= 2.0f;
        val.y *= 2.0f;
        val.z *= 2.0f;
        val.w *= 2.0f;

        // 使用向量存储
        asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                     : : "l"(d_out + idx), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w));

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("向量加载/存储演示:\n");
            printf("  输入：%.2f, %.2f, %.2f, %.2f\n", val.x/2, val.y/2, val.z/2, val.w/2);
            printf("  输出：%.2f, %.2f, %.2f, %.2f\n", val.x, val.y, val.z, val.w);
        }
    }
}

// ============================================================================
// 示例 6: 使用 PTX 实现谓词控制流
// ============================================================================

/**
 * 使用 PTX 谓词控制执行
 * 对应的 PTX 指令：@%p 指令
 * 注意：PTX 谓词需要与 selp 指令配合使用才能获取 u32 结果
 */
__global__ void ptx_predicate_demo(float* d_out, const float* d_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float val = d_in[idx];
        float result;

        // 使用 C++ 比较实现条件执行
        // PTX 原生 setp 指令输出到谓词寄存器，不能直接存储到 u32
        bool is_positive = (val > 0.0f);

        // 根据谓词执行不同操作
        if (is_positive) {
            asm volatile("sqrt.approx.f32 %0, %1;"
                         : "=f"(result)
                         : "f"(val));
        } else {
            asm volatile("abs.f32 %0, %1;"
                         : "=f"(result)
                         : "f"(val));
        }

        d_out[idx] = result;

        if (idx < 4) {
            printf("谓词演示：输入=%.2f, 输出=%.4f, 正数=%s\n",
                   val, result, is_positive ? "是" : "否");
        }
    }
}

// ============================================================================
// 示例 7: 使用 PTX 实现循环展开优化
// ============================================================================

/**
 * 使用 PTX 实现手动循环展开
 * 演示如何通过 PTX 优化循环结构
 */
__global__ void ptx_loop_unroll_demo(float* d_out, const float* d_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    // 手动展开循环 (展开 4 次)
    for (int i = idx; i + 3 < n; i += stride * 4) {
        float v0, v1, v2, v3;

        // 使用 PTX 并行加载
        asm volatile("ld.global.f32 %0, [%1];" : "=f"(v0) : "l"(d_in + i));
        asm volatile("ld.global.f32 %0, [%1];" : "=f"(v1) : "l"(d_in + i + stride));
        asm volatile("ld.global.f32 %0, [%1];" : "=f"(v2) : "l"(d_in + i + stride*2));
        asm volatile("ld.global.f32 %0, [%1];" : "=f"(v3) : "l"(d_in + i + stride*3));

        // 使用 PTX FMA 并行计算
        asm volatile("fma.rn.f32 %0, %1, 2.0, %2;" : "=f"(sum) : "f"(v0), "f"(sum));
        asm volatile("fma.rn.f32 %0, %1, 2.0, %2;" : "=f"(sum) : "f"(v1), "f"(sum));
        asm volatile("fma.rn.f32 %0, %1, 2.0, %2;" : "=f"(sum) : "f"(v2), "f"(sum));
        asm volatile("fma.rn.f32 %0, %1, 2.0, %2;" : "=f"(sum) : "f"(v3), "f"(sum));
    }

    d_out[idx] = sum;

    if (idx == 0) {
        printf("循环展开演示：sum=%.4f\n", sum);
    }
}

// ============================================================================
// 示例 8: 使用 PTX 实现性能计数
// ============================================================================

/**
 * 使用 PTX 性能计数器
 * 对应的 PTX 指令：mov.u64 %rd, %clock64;
 */
__global__ void ptx_performance_demo() {
    int tid = threadIdx.x;

    unsigned long long start_clock = 0;
    unsigned long long end_clock = 0;

    if (tid == 0) {
        // 读取开始时钟
        asm volatile("mov.u64 %0, %clock64;" : "=l"(start_clock));

        // 执行一些计算
        volatile float sum = 0.0f;
        for (int i = 0; i < 1000; i++) {
            sum += i * 0.5f;
        }

        // 读取结束时钟
        asm volatile("mov.u64 %0, %clock64;" : "=l"(end_clock));

        printf("\n性能计数演示:\n");
        printf("  开始时钟：%llu\n", start_clock);
        printf("  结束时钟：%llu\n", end_clock);
        printf("  消耗周期：%llu\n", end_clock - start_clock);
        printf("  迭代次数：1000\n");
        printf("  每迭代周期：%.2f\n", (double)(end_clock - start_clock) / 1000);
    }
}

// ============================================================================
// 示例 9: 使用 PTX 实现 SM ID 查询
// ============================================================================

/**
 * 查询 SM ID
 * 对应的 PTX 指令：mov.u32 %r, %smid;
 */
__global__ void ptx_smid_demo() {
    int smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 8) {
        printf("线程 %d: SM ID = %d\n", idx, smid);
    }
}

// ============================================================================
// 验证函数
// ============================================================================

void verify_atomic_operations() {
    printf("\n----------------------------------------\n");
    printf("  验证原子操作\n");
    printf("----------------------------------------\n");

    unsigned int* d_counter;
    unsigned int* d_result;
    cudaMalloc(&d_counter, sizeof(unsigned int));
    cudaMalloc(&d_result, sizeof(unsigned int));

    // 初始化计数器
    unsigned int initial = 0;
    cudaMemcpy(d_counter, &initial, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // 启动 kernel
    dim3 block(32);
    dim3 grid(1);
    ptx_atomic_demo<<<grid, block>>>(d_counter, d_result);
    cudaDeviceSynchronize();

    cudaFree(d_counter);
    cudaFree(d_result);
}

void verify_vector_operations() {
    printf("\n----------------------------------------\n");
    printf("  验证向量操作\n");
    printf("----------------------------------------\n");

    int n = 16;
    std::vector<float> h_in(n), h_out(n);

    // 初始化输入
    for (int i = 0; i < n; i++) {
        h_in[i] = i * 0.5f;
    }

    // 分配设备内存
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel
    dim3 block(4);
    dim3 grid(1);
    ptx_vector_load_demo<<<grid, block>>>(d_out, d_in, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

void verify_predicate_operations() {
    printf("\n----------------------------------------\n");
    printf("  验证谓词操作\n");
    printf("----------------------------------------\n");

    int n = 8;
    std::vector<float> h_in(n), h_out(n);

    // 初始化输入 (包含正数和负数)
    for (int i = 0; i < n; i++) {
        h_in[i] = (i - 4) * 0.5f;  // -2.0, -1.5, ..., 0, ..., 1.5
    }

    // 分配设备内存
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel
    dim3 block(8);
    dim3 grid(1);
    ptx_predicate_demo<<<grid, block>>>(d_out, d_in, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("========================================\n");
    printf("  PTX 进阶示例代码\n");
    printf("========================================\n");

    // 检查 GPU
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("\nGPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SM 数量：%d\n", prop.multiProcessorCount);

    // 示例 1: 共享内存
    printf("\n----------------------------------------\n");
    printf("  示例 1: PTX 共享内存演示\n");
    printf("----------------------------------------\n");

    int n = 256;
    std::vector<float> h_in(n, 1.0f), h_out(n);
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block1(256);
    dim3 grid1(1);
    ptx_shared_memory_demo<<<grid1, block1>>>(d_out, d_in, n);
    cudaDeviceSynchronize();

    // 示例 2: 原子操作
    printf("\n----------------------------------------\n");
    printf("  示例 2: PTX 原子操作演示\n");
    printf("----------------------------------------\n");

    verify_atomic_operations();

    // 示例 3: 屏障同步
    printf("\n----------------------------------------\n");
    printf("  示例 3: PTX 屏障同步演示\n");
    printf("----------------------------------------\n");

    dim3 block3(32);
    dim3 grid3(1);
    ptx_barrier_demo<<<grid3, block3>>>();
    cudaDeviceSynchronize();

    // 示例 4: 内存屏障
    printf("\n----------------------------------------\n");
    printf("  示例 4: PTX 内存屏障演示\n");
    printf("----------------------------------------\n");

    int *d_data, *d_flag;
    cudaMalloc(&d_data, 2 * sizeof(int));
    cudaMalloc(&d_flag, sizeof(int));
    ptx_membar_demo<<<1, 32>>>(d_data, d_flag);
    cudaDeviceSynchronize();
    cudaFree(d_data);
    cudaFree(d_flag);

    // 示例 5: 向量操作
    printf("\n----------------------------------------\n");
    printf("  示例 5: PTX 向量加载/存储演示\n");
    printf("----------------------------------------\n");

    verify_vector_operations();

    // 示例 6: 谓词控制流
    printf("\n----------------------------------------\n");
    printf("  示例 6: PTX 谓词控制流演示\n");
    printf("----------------------------------------\n");

    verify_predicate_operations();

    // 示例 7: 循环展开
    printf("\n----------------------------------------\n");
    printf("  示例 7: PTX 循环展开演示\n");
    printf("----------------------------------------\n");

    n = 1024;
    h_in.resize(n);
    h_out.resize(n);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    for (int i = 0; i < n; i++) h_in[i] = i * 0.01f;
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block7(64);
    dim3 grid7(4);
    ptx_loop_unroll_demo<<<grid7, block7>>>(d_out, d_in, n);
    cudaDeviceSynchronize();

    // 示例 8: 性能计数
    printf("\n----------------------------------------\n");
    printf("  示例 8: PTX 性能计数演示\n");
    printf("----------------------------------------\n");

    dim3 block8(32);
    dim3 grid8(1);
    ptx_performance_demo<<<grid8, block8>>>();
    cudaDeviceSynchronize();

    // 示例 9: SM ID
    printf("\n----------------------------------------\n");
    printf("  示例 9: PTX SM ID 查询演示\n");
    printf("----------------------------------------\n");

    dim3 block9(32);
    dim3 grid9(prop.multiProcessorCount);
    ptx_smid_demo<<<grid9, block9>>>();
    cudaDeviceSynchronize();

    // 总结
    printf("\n========================================\n");
    printf("  PTX 进阶示例完成!\n");
    printf("========================================\n");
    printf("\n知识点总结:\n");
    printf("1. 共享内存：st.shared, ld.shared\n");
    printf("2. 原子操作：atom.add, atom.cas, atom.min\n");
    printf("3. 屏障同步：bar.sync, membar.gl\n");
    printf("4. 向量操作：ld.global.v4.f32, st.global.v4.f32\n");
    printf("5. 谓词控制：setp.gt.f32, @%p 条件执行\n");
    printf("6. 循环展开：手动展开 4 次迭代\n");
    printf("7. 性能计数：%clock64 寄存器\n");
    printf("8. SM ID 查询：%smid 寄存器\n");

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
