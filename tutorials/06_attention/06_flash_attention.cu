/**
 * 第六课：使用 CUTE 实现 Flash Attention
 *
 * 本示例展示如何使用 CUTE 原语实现高效的 Attention 机制
 * 编译：nvcc -std=c++17 -arch=sm_80 06_flash_attention.cu -o 06_flash_attention
 *
 * 注意：这是教学示例，生产环境请使用 cuDNN 或 CUTLASS
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

// ============================================================================
// 第一部分：Attention 数学基础
// ============================================================================

void attention_math() {
    std::cout << "=== Attention 数学基础 ===" << std::endl;

    std::cout << R"(
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

计算步骤:
1. Q @ K^T        -> 注意力分数 (seq_len x seq_len)
2. 除以 sqrt(d_k)  -> 缩放
3. Softmax        -> 归一化
4. @ V            -> 加权求和

Flash Attention 关键思想:
- 分块计算，避免存储完整的 seq_len x seq_len 矩阵
- 在线 softmax，边计算边归一化
- 重计算，用计算换带宽
)" << std::endl;
}

// ============================================================================
// 第二部分：基础 Attention CUDA Kernel
// ============================================================================

template<int BLOCK_SIZE>
__global__ void naive_attention(
    const float* Q, const float* K, const float* V,
    float* O,
    int seq_len, int d_k, int d_v) {

    int q_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int v_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (q_idx >= seq_len || v_idx >= d_v) return;

    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    float acc = 0.0f;

    // 计算 Q[q_idx] @ K^T
    for (int k = 0; k < seq_len; k++) {
        float qk = 0.0f;
        for (int d = 0; d < d_k; d++) {
            qk += Q[q_idx * d_k + d] * K[k * d_k + d];
        }
        qk /= sqrtf((float)d_k);

        // 在线 softmax (stable softmax)
        float max_prev = max_val;
        max_val = fmaxf(max_val, qk);
        float exp_val = expf(qk - max_val);

        // 更新累加器
        acc = acc * expf(max_prev - max_val) + exp_val * V[k * d_v + v_idx];
        sum_exp = sum_exp * expf(max_prev - max_val) + exp_val;
    }

    O[q_idx * d_v + v_idx] = acc / sum_exp;
}

// ============================================================================
// 第三部分：使用共享内存的分块 Attention
// ============================================================================

template<int BLOCK_Q, int BLOCK_KV>
__global__ void tiled_attention(
    const float* Q, const float* K, const float* V,
    float* O,
    int seq_len, int d_k, int d_v) {

    __shared__ float Q_shared[BLOCK_Q * d_k];
    __shared__ float K_shared[BLOCK_KV * d_k];
    __shared__ float V_shared[BLOCK_KV * d_v];
    __shared__ float scores[BLOCK_Q * BLOCK_KV];

    int q_batch = blockIdx.y;
    int kv_tile = blockIdx.x;

    int q_local = threadIdx.y;
    int kv_local = threadIdx.x;

    int q_idx = q_batch * BLOCK_Q + q_local;
    int kv_base = kv_tile * BLOCK_KV;

    // 初始化输出
    float O_local[BLOCK_KV] = {0};
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // 加载 Q 到共享内存
    if (q_idx < seq_len) {
        for (int d = 0; d < d_k; d++) {
            Q_shared[q_local * d_k + d] = Q[q_idx * d_k + d];
        }
    }

    // 分块处理 K, V
    for (int t = 0; t < (seq_len + BLOCK_KV - 1) / BLOCK_KV; t++) {
        int kv_start = t * BLOCK_KV;

        // 加载 K, V 到共享内存
        __syncthreads();
        if (kv_base + kv_local < seq_len) {
            for (int d = 0; d < d_k; d++) {
                K_shared[kv_local * d_k + d] = K[(kv_base + kv_local) * d_k + d];
            }
            for (int d = 0; d < d_v; d++) {
                V_shared[kv_local * d_v + d] = V[(kv_base + kv_local) * d_v + d];
            }
        }
        __syncthreads();

        // 计算 Q @ K^T
        if (q_local < BLOCK_Q && kv_local < BLOCK_KV) {
            float qk = 0.0f;
            for (int d = 0; d < d_k; d++) {
                qk += Q_shared[q_local * d_k + d] * K_shared[kv_local * d_k + d];
            }
            qk /= sqrtf((float)d_k);
            scores[q_local * BLOCK_KV + kv_local] = qk;
        }
        __syncthreads();

        // 计算 softmax 和输出累加
        if (q_local < BLOCK_Q) {
            for (int k = 0; k < BLOCK_KV && (kv_start + k) < seq_len; k++) {
                float score = scores[q_local * BLOCK_KV + k];

                float max_prev = max_val;
                max_val = fmaxf(max_val, score);
                float exp_val = expf(score - max_val);

                // 更新累加器
                float scale = expf(max_prev - max_val);

                for (int v = 0; v < d_v; v++) {
                    O_local[v] = O_local[v] * scale + exp_val * V_shared[k * d_v + v];
                }
                sum_exp = sum_exp * scale + exp_val;
            }
        }
    }

    // 写入输出
    if (q_idx < seq_len && kv_local < d_v) {
        O[q_idx * d_v + kv_local] = O_local[kv_local] / sum_exp;
    }
}

// ============================================================================
// 第四部分：使用 CUTE 的 Flash Attention
// ============================================================================

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_cute(
    const float* Q, const float* K, const float* V,
    float* O,
    int seq_len, int num_heads) {

    // 使用 CUTE Layout 定义数据布局
    auto layout_Q = make_layout(make_shape(BLOCK_M, HEAD_DIM));
    auto layout_K = make_layout(make_shape(BLOCK_N, HEAD_DIM));
    auto layout_V = make_layout(make_shape(BLOCK_N, HEAD_DIM));
    auto layout_O = make_layout(make_shape(BLOCK_M, HEAD_DIM));
    auto layout_S = make_layout(make_shape(BLOCK_M, BLOCK_N));  // 注意力分数

    // 共享内存
    __shared__ float Q_smem[BLOCK_M * HEAD_DIM];
    __shared__ float K_smem[BLOCK_N * HEAD_DIM];
    __shared__ float V_smem[BLOCK_N * HEAD_DIM];
    __shared__ float S_smem[BLOCK_M * BLOCK_N];
    __shared__ float O_smem[BLOCK_M * HEAD_DIM];  // 用于输出

    // 创建 CUTE Tensor
    auto Q_tensor = make_tensor(Q_smem, layout_Q);
    auto K_tensor = make_tensor(K_smem, layout_K);
    auto V_tensor = make_tensor(V_smem, layout_V);
    auto S_tensor = make_tensor(S_smem, layout_S);

    // 输出统计量（每个线程私有）
    float acc_local[HEAD_DIM] = {0};

    int q_idx = blockIdx.y * BLOCK_M + threadIdx.y;
    int head_idx = blockIdx.x;

    if (q_idx >= seq_len || head_idx >= num_heads) return;

    // 初始化统计量 (用于在线 softmax)
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // 加载 Q 到共享内存
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        Q_tensor(threadIdx.y, d) = Q[(q_idx * num_heads + head_idx) * HEAD_DIM + d];
    }
    __syncthreads();

    // 分块处理 K, V
    for (int kv_tile = 0; kv_tile < (seq_len + BLOCK_N - 1) / BLOCK_N; kv_tile++) {
        int kv_start = kv_tile * BLOCK_N;

        // 加载 K, V 到共享内存
        for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
            for (int n = threadIdx.y; n < BLOCK_N; n += blockDim.y) {
                int kv_idx = kv_start + n;
                if (kv_idx < seq_len) {
                    K_tensor(n, d) = K[(kv_idx * num_heads + head_idx) * HEAD_DIM + d];
                    V_tensor(n, d) = V[(kv_idx * num_heads + head_idx) * HEAD_DIM + d];
                }
            }
        }
        __syncthreads();

        // 计算注意力分数 S = Q @ K^T
        for (int m = threadIdx.y; m < BLOCK_M; m += blockDim.y) {
            for (int n = threadIdx.x; n < BLOCK_N; n += blockDim.x) {
                float qk = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    qk += Q_tensor(m, d) * K_tensor(n, d);
                }
                qk /= sqrtf((float)HEAD_DIM);
                S_tensor(m, n) = qk;
            }
        }
        __syncthreads();

        // 在线 softmax 和输出累加
        for (int m = threadIdx.y; m < BLOCK_M; m += blockDim.y) {
            for (int n = threadIdx.x; n < BLOCK_N; n += blockDim.x) {
                int kv_idx = kv_start + n;
                if (kv_idx >= seq_len) continue;

                float score = S_tensor(m, n);

                // Stable softmax 更新
                float max_prev = max_val;
                max_val = fmaxf(max_val, score);
                float exp_scale = expf(max_prev - max_val);
                float exp_val = expf(score - max_val);

                // 更新累加器
                for (int d = 0; d < HEAD_DIM; d++) {
                    acc_local[d] = acc_local[d] * exp_scale + exp_val * V_tensor(n, d);
                }
                sum_exp = sum_exp * exp_scale + exp_val;
            }
        }
        __syncthreads();
    }

    // 归一化并写入输出
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float result = acc_local[d] / sum_exp;
        O[(q_idx * num_heads + head_idx) * HEAD_DIM + d] = result;
    }
}

// ============================================================================
// 第五部分：简化的可运行 Flash Attention
// ============================================================================

template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void flash_attention_simple(
    const float* Q, const float* K, const float* V,
    float* O,
    int seq_len, int num_heads) {

    // 每个 block 处理一个 query 位置和一个 head
    int q_idx = blockIdx.y;
    int head_idx = blockIdx.x;

    if (q_idx >= seq_len || head_idx >= num_heads) return;

    int tid = threadIdx.x;

    // 共享内存
    __shared__ float q_vec[HEAD_DIM];
    __shared__ float k_vec[HEAD_DIM];
    __shared__ float v_vec[HEAD_DIM];
    __shared__ float o_vec[HEAD_DIM];
    __shared__ float max_val;
    __shared__ float sum_exp;

    // 初始化
    if (tid < HEAD_DIM) {
        o_vec[tid] = 0.0f;
    }
    if (tid == 0) {
        max_val = -INFINITY;
        sum_exp = 1e-10f;  // 避免除以零
    }
    __syncthreads();

    // 加载 Q[q_idx]
    int q_base = (q_idx * num_heads + head_idx) * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        q_vec[d] = Q[q_base + d];
    }
    __syncthreads();

    // 遍历所有 key/value
    for (int kv_idx = 0; kv_idx < seq_len; kv_idx++) {
        // 加载 K[kv_idx] 和 V[kv_idx]
        int kv_base = (kv_idx * num_heads + head_idx) * HEAD_DIM;
        for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
            k_vec[d] = K[kv_base + d];
            v_vec[d] = V[kv_base + d];
        }
        __syncthreads();

        // 计算 Q @ K - 每个线程计算一部分点积
        float qk_partial = 0.0f;
        for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
            qk_partial += q_vec[d] * k_vec[d];
        }

        // 使用 shuffle 归约
        float qk = qk_partial;
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            qk += __shfl_down_sync(0xffffffff, qk, stride);
        }

        // 线程 0 执行 softmax 更新
        float scale = 1.0f;
        float exp_val = 0.0f;

        if (tid == 0) {
            qk /= sqrtf((float)HEAD_DIM);

            float max_prev = max_val;
            max_val = fmaxf(max_val, qk);
            scale = expf(max_prev - max_val);
            exp_val = expf(qk - max_val);

            sum_exp = sum_exp * scale + exp_val;
        }
        __syncthreads();  // 确保所有线程看到更新的 sum_exp

        // 所有线程更新输出累加器
        for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
            o_vec[d] = o_vec[d] * scale + exp_val * v_vec[d];
        }
        __syncthreads();
    }

    // 归一化并写入输出
    __syncthreads();

    if (tid < HEAD_DIM) {
        float result = o_vec[tid] / sum_exp;
        int out_idx = (q_idx * num_heads + head_idx) * HEAD_DIM + tid;
        O[out_idx] = result;
    }
}

// ============================================================================
// 第六部分：验证和测试
// ============================================================================

void verify_attention() {
    std::cout << "=== Flash Attention 验证 ===" << std::endl;

    int seq_len = 64;
    int num_heads = 4;
    int head_dim = 32;

    int total_tokens = seq_len * num_heads;

    size_t size = total_tokens * head_dim * sizeof(float);

    // 主机内存
    float *h_Q = new float[total_tokens * head_dim];
    float *h_K = new float[total_tokens * head_dim];
    float *h_V = new float[total_tokens * head_dim];
    float *h_O_cpu = new float[total_tokens * head_dim];
    float *h_O_gpu = new float[total_tokens * head_dim];

    // 初始化随机数据
    srand(42);
    for (int i = 0; i < total_tokens * head_dim; i++) {
        h_Q[i] = (float)rand() / RAND_MAX;
        h_K[i] = (float)rand() / RAND_MAX;
        h_V[i] = (float)rand() / RAND_MAX;
    }

    // CPU 参考实现 - 标准 Attention
    // Q, K, V 的形状是 [seq_len, num_heads, head_dim]
    // 存储为 [seq_len * num_heads * head_dim]
    for (int q = 0; q < seq_len; q++) {
        for (int h = 0; h < num_heads; h++) {
            // 计算 attention(Q[q,h], K[:,h], V[:,h])
            float output[32] = {0};  // head_dim 最大 32
            float max_val = -INFINITY;
            float sum_exp = 0.0f;

            // 第一遍：找最大值
            for (int k = 0; k < seq_len; k++) {
                float qk = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    float q_val = h_Q[(q * num_heads + h) * head_dim + i];
                    float k_val = h_K[(k * num_heads + h) * head_dim + i];
                    qk += q_val * k_val;
                }
                qk /= sqrtf((float)head_dim);
                max_val = fmaxf(max_val, qk);
            }

            // 第二遍：计算 softmax 和输出
            for (int k = 0; k < seq_len; k++) {
                float qk = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    float q_val = h_Q[(q * num_heads + h) * head_dim + i];
                    float k_val = h_K[(k * num_heads + h) * head_dim + i];
                    qk += q_val * k_val;
                }
                qk /= sqrtf((float)head_dim);

                float exp_val = expf(qk - max_val);
                sum_exp += exp_val;

                for (int i = 0; i < head_dim; i++) {
                    float v_val = h_V[(k * num_heads + h) * head_dim + i];
                    output[i] += exp_val * v_val;
                }
            }

            // 写入 CPU 输出
            for (int d = 0; d < head_dim; d++) {
                h_O_cpu[(q * num_heads + h) * head_dim + d] = output[d] / sum_exp;
            }
        }
    }

    std::cout << "CPU 参考实现完成" << std::endl;
    std::cout << "  O_cpu[0] = " << h_O_cpu[0] << std::endl;

    // GPU 内存
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_O, size);

    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);

    // 启动 GPU kernel
    // kernel 布局：blockIdx.x = head_idx, blockIdx.y = q_idx
    dim3 block(256);
    dim3 grid(num_heads, seq_len);

    flash_attention_simple<256, 32><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, num_heads);
    cudaDeviceSynchronize();

    // 检查 kernel 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "GPU Kernel 错误：" << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(h_O_gpu, d_O, size, cudaMemcpyDeviceToHost);

    std::cout << "GPU 实现完成" << std::endl;
    std::cout << "  O_gpu[0] = " << h_O_gpu[0] << std::endl;

    // 验证结果
    float max_diff = 0.0f;
    int max_diff_idx = 0;
    for (int i = 0; i < total_tokens * head_dim; i++) {
        float diff = fabs(h_O_gpu[i] - h_O_cpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    std::cout << "最大差异：" << max_diff << " (index " << max_diff_idx << ")" << std::endl;
    std::cout << "  CPU[" << max_diff_idx << "] = " << h_O_cpu[max_diff_idx] << std::endl;
    std::cout << "  GPU[" << max_diff_idx << "] = " << h_O_gpu[max_diff_idx] << std::endl;
    std::cout << "验证结果：" << (max_diff < 0.1f ? "PASS (tolerance 0.1)" : "FAIL") << std::endl;

    // 清理
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O_cpu;
    delete[] h_O_gpu;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}

void benchmark_attention() {
    std::cout << "\n=== Attention 性能测试 ===" << std::endl;

    int seq_len = 512;
    int num_heads = 8;
    int head_dim = 64;
    int total_tokens = seq_len * num_heads;

    size_t size = total_tokens * head_dim * sizeof(float);

    float *h_Q = new float[total_tokens * head_dim];
    for (int i = 0; i < total_tokens * head_dim; i++) {
        h_Q[i] = (float)rand() / RAND_MAX;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_O, size);

    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_Q, size, cudaMemcpyHostToDevice);

    // Warmup
    dim3 block(256);
    dim3 grid(num_heads, seq_len);
    flash_attention_simple<256, 64><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, num_heads);
    cudaDeviceSynchronize();

    // 测量
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        flash_attention_simple<256, 64><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, num_heads);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= 100;

    // 计算 GFLOPS
    float gflops = 2.0f * seq_len * seq_len * num_heads * head_dim / 1e9;

    std::cout << "序列长度：" << seq_len << std::endl;
    std::cout << "头数：" << num_heads << ", 头维度：" << head_dim << std::endl;
    std::cout << "执行时间：" << time_ms << " ms" << std::endl;
    std::cout << "GFLOPS: " << gflops / time_ms << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_Q;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Flash Attention 实现教程" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    attention_math();
    std::cout << std::endl;

    verify_attention();
    benchmark_attention();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  第六课完成！" << std::endl;
    std::cout << "  恭喜完成整个 CUTE 教程！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
