/**
 * 第十三课：Flash Attention v2 实现
 *
 * 本课讲解 Flash Attention v2 的核心优化：
 * 1. 并行化策略
 * 2. 重计算技术
 * 3. 在线 softmax 优化
 *
 * 编译：nvcc -std=c++17 -arch=sm_80 13_attention_v2.cu -o 13_attention_v2
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// ============================================================================
// Flash Attention v2 原理
// ============================================================================

void test_flash_attn_v2_principles() {
    std::cout << "=== Flash Attention v2 原理 ===" << std::endl;

    std::cout << R"(
Flash Attention v2 优化点:

1. 减少非矩阵乘法操作:
   - 重新安排计算顺序
   - 减少 softmax 开销

2. 改进的并行化:
   - 序列维度并行
   - 头维度并行

3. 重计算:
   - 用计算换带宽
   - 避免存储大矩阵

Attention 公式:
  Attention(Q, K, V) = softmax(QK^T / √d) V

计算步骤:
  1. 分块加载 Q, K, V
  2. 计算 QK^T (分块)
  3. 在线 softmax
  4. 累加 V
    )" << std::endl;
}

// ============================================================================
// Flash Attention v2 Kernel
// ============================================================================

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_v2_kernel(
    const float* Q, const float* K, const float* V,
    float* O, float* L, float* M,  // L=softmax denominator, M=max
    int seq_len, int num_heads) {

    // 每个 block 处理一个 head 的一部分
    int head_idx = blockIdx.x;
    int m_block = blockIdx.y;

    if (head_idx >= num_heads || m_block * BLOCK_M >= seq_len) return;

    int tid = threadIdx.x;

    // 共享内存
    __shared__ float Q_shared[BLOCK_M * HEAD_DIM];
    __shared__ float K_shared[BLOCK_N * HEAD_DIM];
    __shared__ float V_shared[BLOCK_N * HEAD_DIM];
    __shared__ float O_local[BLOCK_M * HEAD_DIM];
    __shared__ float m_local[BLOCK_M];  // per-row max
    __shared__ float l_local[BLOCK_M];  // per-row sum

    // 初始化
    int q_start = m_block * BLOCK_M;
    int q_idx = q_start + tid / HEAD_DIM;
    int d_idx = tid % HEAD_DIM;

    if (q_idx < seq_len && d_idx < HEAD_DIM) {
        int idx = ((q_idx * num_heads + head_idx) * HEAD_DIM) + d_idx;
        Q_shared[(q_idx - q_start) * HEAD_DIM + d_idx] = Q[idx];
        O_local[(q_idx - q_start) * HEAD_DIM + d_idx] = 0.0f;
    }
    if (tid < BLOCK_M) {
        m_local[tid] = -INFINITY;
        l_local[tid] = 0.0f;
    }
    __syncthreads();

    // 遍历 K, V 块
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    for (int n_block = 0; n_block < (seq_len + BLOCK_N - 1) / BLOCK_N; n_block++) {
        int kv_start = n_block * BLOCK_N;

        // 加载 K, V
        int kv_idx = kv_start + tid / HEAD_DIM;
        int d_idx_kv = tid % HEAD_DIM;
        if (kv_idx < seq_len && d_idx_kv < HEAD_DIM) {
            int idx = ((kv_idx * num_heads + head_idx) * HEAD_DIM) + d_idx_kv;
            K_shared[(kv_idx - kv_start) * HEAD_DIM + d_idx_kv] = K[idx];
            V_shared[(kv_idx - kv_start) * HEAD_DIM + d_idx_kv] = V[idx];
        }
        __syncthreads();

        // 计算 attention scores
        for (int i = tid; i < BLOCK_M; i += blockDim.x) {
            int q_row = q_start + i;
            if (q_row >= seq_len) continue;

            float max_prev = m_local[i];
            float sum_prev = l_local[i];

            // 计算 Q[i] · K[j] for all j in block
            for (int j = 0; j < BLOCK_N && (kv_start + j) < seq_len; j++) {
                float qk = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    qk += Q_shared[i * HEAD_DIM + d] * K_shared[j * HEAD_DIM + d];
                }
                qk *= scale;

                // Online softmax
                float m_new = fmaxf(max_prev, qk);
                float exp_mk = expf(qk - m_new);
                float exp_mprev = expf(max_prev - m_new);

                // Update sum
                float sum_new = sum_prev * exp_mprev + exp_mk;

                // Update output
                for (int d = 0; d < HEAD_DIM; d++) {
                    O_local[i * HEAD_DIM + d] =
                        (O_local[i * HEAD_DIM + d] * sum_prev * exp_mprev +
                         exp_mk * V_shared[j * HEAD_DIM + d]) / sum_new;
                }

                max_prev = m_new;
                sum_prev = sum_new;
            }

            m_local[i] = max_prev;
            l_local[i] = sum_prev;
        }
        __syncthreads();
    }

    // 写回结果
    if (q_idx < seq_len && d_idx < HEAD_DIM) {
        int out_idx = ((q_idx * num_heads + head_idx) * HEAD_DIM) + d_idx;
        O[out_idx] = O_local[(q_idx - q_start) * HEAD_DIM + d_idx];
    }

    // 保存统计量（用于 backward）
    if (tid < BLOCK_M && (q_start + tid) < seq_len) {
        L[(m_block * num_heads + head_idx) * BLOCK_M + tid] = l_local[tid];
        M[(m_block * num_heads + head_idx) * BLOCK_M + tid] = m_local[tid];
    }
}

// ============================================================================
// 简化的 Flash Attention (用于演示)
// ============================================================================

template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void flash_attention_simple(
    const float* Q, const float* K, const float* V,
    float* O,
    int seq_len, int num_heads) {

    int head_idx = blockIdx.x;
    int q_idx = blockIdx.y;

    if (head_idx >= num_heads || q_idx >= seq_len) return;

    int tid = threadIdx.x;

    // 共享内存
    __shared__ float q_vec[HEAD_DIM];
    __shared__ float k_vec[HEAD_DIM];
    __shared__ float v_vec[HEAD_DIM];
    __shared__ float o_vec[HEAD_DIM];

    // 初始化
    if (tid < HEAD_DIM) {
        o_vec[tid] = 0.0f;
    }

    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    __syncthreads();

    // 加载 Q
    int q_base = (q_idx * num_heads + head_idx) * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        q_vec[d] = Q[q_base + d];
    }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    // 遍历所有 K, V
    for (int kv_idx = 0; kv_idx < seq_len; kv_idx++) {
        // 加载 K, V
        int kv_base = (kv_idx * num_heads + head_idx) * HEAD_DIM;
        for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
            k_vec[d] = K[kv_base + d];
            v_vec[d] = V[kv_base + d];
        }
        __syncthreads();

        // 计算 Q · K
        float qk = 0.0f;
        for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
            qk += q_vec[d] * k_vec[d];
        }

        // Warp 归约
        for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
            qk += __shfl_down_sync(0xffffffff, qk, s);
        }

        // 更新 softmax 统计量
        if (tid == 0) {
            qk *= scale;
            float max_prev = max_val;
            max_val = fmaxf(max_prev, qk);
            float exp_scale = expf(max_prev - max_val);
            float exp_val = expf(qk - max_val);

            // 更新输出
            for (int d = 0; d < HEAD_DIM; d++) {
                o_vec[d] = o_vec[d] * exp_scale + exp_val * v_vec[d];
            }
            sum_exp = sum_exp * exp_scale + exp_val;
        }
        __syncthreads();
    }

    // 归一化并输出
    if (tid < HEAD_DIM) {
        int out_idx = (q_idx * num_heads + head_idx) * HEAD_DIM + tid;
        O[out_idx] = o_vec[tid] / sum_exp;
    }
}

void test_flash_attention_v2() {
    std::cout << "=== Flash Attention v2 测试 ===" << std::endl;

    int seq_len = 64;
    int num_heads = 4;
    int head_dim = 32;

    size_t size = seq_len * num_heads * head_dim * sizeof(float);

    // 分配内存
    float *h_Q = new float[seq_len * num_heads * head_dim];
    float *h_K = new float[seq_len * num_heads * head_dim];
    float *h_V = new float[seq_len * num_heads * head_dim];
    float *h_O = new float[seq_len * num_heads * head_dim];

    // 初始化
    for (int i = 0; i < seq_len * num_heads * head_dim; i++) {
        h_Q[i] = 0.1f;
        h_K[i] = 0.1f;
        h_V[i] = 0.1f;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_O, size);

    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);

    // 启动 Kernel
    dim3 block(256);
    dim3 grid(num_heads, seq_len);

    flash_attention_simple<256, 32><<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, num_heads);
    cudaDeviceSynchronize();

    cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost);

    std::cout << "Flash Attention v2 结果:" << std::endl;
    std::cout << "  O[0] = " << h_O[0] << std::endl;
    std::cout << "  (期望约 0.1)" << std::endl;

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}

// ============================================================================
// 性能对比
// ============================================================================

void test_performance_comparison() {
    std::cout << "\n=== 性能对比 ===" << std::endl;

    std::cout << "Flash Attention vs Standard Attention:" << std::endl;
    std::cout << "\nStandard Attention:" << std::endl;
    std::cout << "  内存：O(N²) 存储 attention matrix" << std::endl;
    std::cout << "  带宽：多次全局内存访问" << std::endl;

    std::cout << "\nFlash Attention:" << std::endl;
    std::cout << "  内存：O(N) 只存输出" << std::endl;
    std::cout << "  带宽：SRAM 优化，减少 HBM 访问" << std::endl;
    std::cout << "  加速：2-3x (长序列)" << std::endl;

    std::cout << "\nFlash Attention v2 改进:" << std::endl;
    std::cout << "  - 减少非 MMA 操作" << std::endl;
    std::cout << "  - 改进并行化" << std::endl;
    std::cout << "  - 更好的 occupancy" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  第十三课：Flash Attention v2" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_flash_attn_v2_principles();
    test_flash_attention_v2();
    test_performance_comparison();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  第十三课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
