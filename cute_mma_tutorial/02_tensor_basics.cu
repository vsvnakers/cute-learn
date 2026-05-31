/**
 * ============================================================================
 * CuTe + MMA 教程 02: Tensor 基础
 * ============================================================================
 *
 * Tensor = Engine（数据引擎） + Layout（布局）
 *
 * Engine 负责存储数据（可以是普通指针、寄存器数组等）
 * Layout 负责描述如何通过多维坐标访问这些数据
 *
 * CuTe 的 Tensor 是 view（视图），不拥有数据本身。
 * 类似于 std::span 或 std::mdspan。
 *
 * 编译：make 02_tensor_basics
 * 运行：./02_tensor_basics
 */

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

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
// 1. Host 端 Tensor
// ============================================================================
// make_tensor 是创建 Tensor 的核心 API
// 它接受一个数据指针和一个 Layout，返回一个 Tensor view
//
// 重要：CuTe 的 make_tensor 需要指针类型（如 float*），不能直接用 C 数组名
// C 数组名会被推导为数组类型而非指针

void test_host_tensor() {
    std::cout << "=== 1. Host 端 Tensor ===" << std::endl;
    std::cout << std::endl;

    // ---- 创建一维 Tensor ----
    float data_1d[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    // 必须用指针：make_tensor(ptr, layout)
    // ptr 的类型是 float*，CuTe 用它作为 Engine
    auto tensor_1d = make_tensor(static_cast<float*>(data_1d),
                                 make_layout(make_shape(Int<8>{})));
    std::cout << "  1D Tensor:" << std::endl;
    std::cout << "    Layout = " << tensor_1d.layout() << std::endl;
    std::cout << "    数据 = ";
    for (int i = 0; i < 8; i++) {
        std::cout << tensor_1d(i) << " ";  // 使用 () 运算符访问
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // ---- 创建二维 Tensor (Row-Major) ----
    // 默认 Row-Major: stride = (N, 1)
    // CuTe 默认的 make_layout 是 Column-Major!
    // 需要手动指定 Row-Major stride
    float data_2d[12] = {0,1,2,3, 4,5,6,7, 8,9,10,11};  // 3x4 矩阵
    auto tensor_2d = make_tensor(static_cast<float*>(data_2d),
                                 make_layout(make_shape(Int<3>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})));
    std::cout << "  2D Tensor (3x4, Row-Major stride=(4,1)):" << std::endl;
    std::cout << "    Layout = " << tensor_2d.layout() << std::endl;
    std::cout << "    数据:" << std::endl;
    for (int m = 0; m < 3; m++) {
        std::cout << "      ";
        for (int n = 0; n < 4; n++) {
            std::cout << std::setw(4) << tensor_2d(m, n);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ---- 创建二维 Tensor (默认 stride) ----
    // CuTe 默认 stride: make_layout(shape) -> Column-Major (stride=(1,M))
    auto tensor_2d_default = make_tensor(static_cast<float*>(data_2d),
                                         make_layout(make_shape(Int<3>{}, Int<4>{})));
    std::cout << "  2D Tensor (3x4, 默认 stride):" << std::endl;
    std::cout << "    Layout = " << tensor_2d_default.layout() << std::endl;
    std::cout << "    注意: CuTe 默认 stride 是 Column-Major (1,3)" << std::endl;
    std::cout << "    数据 (按 (m,n) 访问):" << std::endl;
    for (int m = 0; m < 3; m++) {
        std::cout << "      ";
        for (int n = 0; n < 4; n++) {
            std::cout << std::setw(4) << tensor_2d_default(m, n);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// ============================================================================
// 2. Device 端 Tensor
// ============================================================================
// CUDA kernel 中的 Tensor 访问方式与 host 端相同
// CuTe 的 Tensor 是轻量级 view，可以安全地在 host/device 间传递

__global__ void device_tensor_demo(float* data, int M, int N) {
    // 使用运行时 shape 创建 device Tensor
    auto tensor = make_tensor(data, make_layout(make_shape(M, N)));

    // 每个线程处理一个元素
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int m = idx / N;
        int n = idx % N;
        // 修改数据，验证 device 端访问
        tensor(m, n) = tensor(m, n) * 2.0f;
    }
}

void test_device_tensor() {
    std::cout << "=== 2. Device 端 Tensor ===" << std::endl;
    std::cout << std::endl;

    const int M = 3, N = 4;
    const int size = M * N;

    // Host 数据
    float h_data[12] = {0,1,2,3, 4,5,6,7, 8,9,10,11};

    // Device 分配
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice));

    // 启动 kernel
    device_tensor_demo<<<1, size>>>(d_data, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷贝回 host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "  Device Tensor 修改后 (所有元素 *2):" << std::endl;
    for (int m = 0; m < M; m++) {
        std::cout << "      ";
        for (int n = 0; n < N; n++) {
            std::cout << std::setw(6) << h_data[m * N + n];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_data));
}

// ============================================================================
// 3. Tensor 的切片 (Slicing)
// ============================================================================
// CuTe 支持对 Tensor 进行切片，返回子 Tensor view
// 这在 MMA 中非常重要：将大 Tensor 切片为每个线程负责的部分

void test_tensor_slice() {
    std::cout << "=== 3. Tensor 切片 ===" << std::endl;
    std::cout << std::endl;

    float data[24];
    for (int i = 0; i < 24; i++) data[i] = (float)i;

    // 创建 4x6 Tensor (Column-Major 默认)
    auto tensor = make_tensor(static_cast<float*>(data),
                              make_layout(make_shape(Int<4>{}, Int<6>{})));
    std::cout << "  原始 Tensor (4x6):" << std::endl;
    for (int m = 0; m < 4; m++) {
        std::cout << "      ";
        for (int n = 0; n < 6; n++) {
            std::cout << std::setw(4) << tensor(m, n);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ---- 行切片 ----
    // tensor(row_idx, _) 获取第 row_idx 行
    auto row0 = tensor(Int<0>{}, _);  // 第 0 行，编译期索引
    auto row1 = tensor(1, _);          // 第 1 行，运行时索引
    std::cout << "  tensor(0, _) = " << row0 << std::endl;
    std::cout << "  tensor(1, _) = " << row1 << std::endl;

    // ---- 列切片 ----
    // tensor(_, col_idx) 获取第 col_idx 列
    auto col2 = tensor(_, Int<2>{});  // 第 2 列
    std::cout << "  tensor(_, 2) = " << col2 << std::endl;

    // ---- 单元素访问 ----
    // tensor(m, n) 访问单个元素
    std::cout << "  tensor(2, 3) = " << tensor(2, 3) << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 4. Tensor 的 reshape
// ============================================================================
// CuTe 的 Tensor 支持 reshape（改变形状但不改变数据）
// 这在 MMA 中用于将线程的 flat view 转换为 2D view

void test_tensor_reshape() {
    std::cout << "=== 4. Tensor reshape ===" << std::endl;
    std::cout << std::endl;

    float data[12] = {0,1,2,3,4,5,6,7,8,9,10,11};

    // 原始 1D Tensor
    auto flat = make_tensor(static_cast<float*>(data),
                            make_layout(make_shape(Int<12>{})));
    std::cout << "  1D Tensor: " << flat.layout() << std::endl;

    // reshape 为 3x4
    auto matrix = make_tensor(static_cast<float*>(data),
                              make_layout(make_shape(Int<3>{}, Int<4>{})));
    std::cout << "  3x4 Tensor: " << matrix.layout() << std::endl;

    // reshape 为 (2,2,3)
    auto tensor_3d = make_tensor(static_cast<float*>(data),
                                 make_layout(make_shape(Int<2>{}, Int<2>{}, Int<3>{})));
    std::cout << "  (2,2,3) Tensor: " << tensor_3d.layout() << std::endl;

    // 验证数据一致性
    std::cout << std::endl;
    std::cout << "  验证数据一致性 (同一底层数据):" << std::endl;
    std::cout << "    flat(5) = " << flat(5) << std::endl;
    std::cout << "    matrix(1,1) = " << matrix(1, 1) << std::endl;
    std::cout << "    tensor_3d(0,1,2) = " << tensor_3d(0, 1, 2) << std::endl;
    std::cout << "    三者都指向 data[5] = 5" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// 5. Tensor 的常用属性
// ============================================================================

void test_tensor_properties() {
    std::cout << "=== 5. Tensor 常用属性 ===" << std::endl;
    std::cout << std::endl;

    float data[12];
    auto tensor = make_tensor(static_cast<float*>(data),
                              make_layout(make_shape(Int<3>{}, Int<4>{})));

    // shape: 形状
    std::cout << "  shape(tensor)  = " << shape(tensor) << std::endl;

    // layout: 布局
    std::cout << "  layout(tensor) = " << tensor.layout() << std::endl;

    // size: 总元素数
    std::cout << "  size(tensor)   = " << size(tensor) << std::endl;

    // rank: 维度数
    std::cout << "  rank(tensor)   = " << rank(tensor) << std::endl;

    // data(): 获取底层数据指针
    std::cout << "  data() 指针    = " << tensor.data() << std::endl;
    std::cout << "  data 地址      = " << (void*)data << std::endl;

    // is_unique: 通过 layout 的 size 判断是否一一映射
    // size(layout) == size(shape(layout)) 时为 unique
    std::cout << "  size(layout)   = " << size(tensor.layout()) << std::endl;
    std::cout << "  size(shape)    = " << size(shape(tensor)) << std::endl;

    std::cout << std::endl;
}

// ============================================================================
// 6. Shared Memory Tensor
// ============================================================================
// 在 CUDA kernel 中，shared memory 是 MMA 的关键
// CuTe 可以直接为 shared memory 创建 Tensor

__global__ void smem_tensor_demo() {
    // 声明 shared memory
    __shared__ float smem[16 * 8];

    // 为 shared memory 创建 Tensor
    // 使用 make_smem_ptr 标记为 shared memory 指针
    auto smem_tensor = make_tensor(
        make_smem_ptr(smem),
        make_layout(make_shape(Int<16>{}, Int<8>{}))
    );

    // 每个线程填充一个元素
    int tid = threadIdx.x;
    if (tid < 16 * 8) {
        int m = tid / 8;
        int n = tid % 8;
        smem_tensor(m, n) = (float)(m * 8 + n);
    }
    __syncthreads();

    // 验证
    if (tid == 0) {
        printf("  smem_tensor(0,0) = %.1f\n", smem_tensor(0, 0));
        printf("  smem_tensor(1,0) = %.1f\n", smem_tensor(1, 0));
        printf("  smem_tensor(15,7) = %.1f\n", smem_tensor(15, 7));
    }
}

void test_smem_tensor() {
    std::cout << "=== 6. Shared Memory Tensor ===" << std::endl;
    std::cout << std::endl;

    // 启动 kernel
    smem_tensor_demo<<<1, 128>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << std::endl;
}

// ============================================================================
// 7. Register Tensor (寄存器 Tensor)
// ============================================================================
// MMA 操作在寄存器上进行
// 每个线程持有 Tensor 的一个 fragment（片段）

__global__ void register_tensor_demo() {
    // 每个线程持有 4 个 float 作为寄存器 fragment
    float reg[4];
    for (int i = 0; i < 4; i++) reg[i] = (float)(threadIdx.x * 4 + i);

    // 创建寄存器 Tensor
    // 使用寄存器数组作为底层存储，需要转换为指针
    auto frag = make_tensor(static_cast<float*>(reg),
                            make_layout(make_shape(Int<2>{}, Int<2>{})));

    // frag 是一个 2x2 的小 Tensor
    if (threadIdx.x == 0) {
        printf("  Thread 0 fragment (2x2):\n");
        for (int m = 0; m < 2; m++) {
            printf("    ");
            for (int n = 0; n < 2; n++) {
                printf("%6.1f", frag(m, n));
            }
            printf("\n");
        }
    }
}

void test_register_tensor() {
    std::cout << "=== 7. Register Tensor ===" << std::endl;
    std::cout << std::endl;

    register_tensor_demo<<<1, 32>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CuTe + MMA 教程 02: Tensor 基础" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_host_tensor();
    test_device_tensor();
    test_tensor_slice();
    test_tensor_reshape();
    test_tensor_properties();
    test_smem_tensor();
    test_register_tensor();

    std::cout << "========================================" << std::endl;
    std::cout << "  教程 02 完成！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
