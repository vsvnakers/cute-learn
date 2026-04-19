/**
 * 第十课：Cluster 和 Warp 同步
 *
 * 本课讲解 Hopper 架构的 Cluster 和 Warp 级同步原语：
 * 1. Cluster 概念
 * 2. Warp Group
 * 3. 同步机制
 *
 * 编译：nvcc -std=c++17 -arch=sm_90 10_cluster_warp.cu -o 10_cluster_warp
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cute/layout.hpp>

using namespace cute;

void test_cluster_basics() {
    std::cout << "=== Cluster 基础 ===" << std::endl;

    std::cout << R"(
Cluster 是 Hopper 引入的新概念:

传统层次:
  Grid -> Block -> Warp -> Thread

Hopper 层次:
  Grid -> Cluster -> CTA (Block) -> Warp Group -> Warp -> Thread

Cluster 特点:
  - 多个 CTA 组成一个 Cluster
  - Cluster 内 CTA 可以高效同步
  - 支持 Cluster 范围的屏障
  - 共享内存可以跨 CTA 访问
    )" << std::endl;
}

void test_warp_group() {
    std::cout << "\n=== Warp Group ===" << std::endl;

    // Warp Group 是 Hopper 的新特性
    std::cout << "Warp Group (Hopper):" << std::endl;
    std::cout << "  - 一个 Warp Group = 32 线程" << std::endl;
    std::cout << "  - wgmma 在 Warp Group 级别执行" << std::endl;
    std::cout << "  - 比传统 Warp 更高效" << std::endl;

    // Warp 级同步
    std::cout << "\nWarp 级同步原语:" << std::endl;
    std::cout << "  - __syncwarp(): Warp 内同步" << std::endl;
    std::cout << "  - __shfl_sync(): Warp 内数据交换" << std::endl;
    std::cout << "  - __ballot_sync(): Warp 投票" << std::endl;
}

void test_barrier() {
    std::cout << "\n=== 屏障同步 ===" << std::endl;

    // 不同级别的屏障
    std::cout << "屏障同步级别:" << std::endl;
    std::cout << "  Thread 级:  atomicAdd, fence" << std::endl;
    std::cout << "  Warp 级：   __syncwarp()" << std::endl;
    std::cout << "  Block 级：  __syncthreads()" << std::endl;
    std::cout << "  Cluster 级：cudaClusterBarrier (Hopper)" << std::endl;

    // mbarrier
    std::cout << "\nmbarrier (Hopper):" << std::endl;
    std::cout << "  - 硬件加速屏障" << std::endl;
    std::cout << "  - 支持 wait 和 arrive 操作" << std::endl;
    std::cout << "  - 用于 TMA 同步" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  第十课：Cluster 和 Warp 同步" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_cluster_basics();
    test_warp_group();
    test_barrier();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  注意：Cluster 需要 Hopper (SM90) GPU" << std::endl;
    std::cout << "  第十课完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
