cmake_minimum_required(VERSION 3.18)
project(cute_tutorials LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# 设置 CUDA 架构 - 支持 RTX 3060 (SM86) 和 A100 (SM80)
set(CMAKE_CUDA_ARCHITECTURES 80 86)

# 启用更多警告
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")

# 找到 CUTLASS
set(CUTLASS_PATH "${CMAKE_SOURCE_DIR}/../cutlass")
include_directories(${CUTLASS_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================
# 基础教程 (1-6)
# ============================================
add_executable(01_layout_basic 01_layout/01_layout_basic.cu)
add_executable(02_swizzle_basic 02_swizzle/02_swizzle_basic.cu)
add_executable(03_bank_conflict 03_bank_conflict/03_bank_conflict.cu)
add_executable(04_mma_basic 04_mma/04_mma_basic.cu)
add_executable(05_gemm 05_gemm/05_gemm.cu)
add_executable(06_flash_attention 06_attention/06_flash_attention.cu)

# ============================================
# 中级教程 (7-10)
# ============================================

# 07: Tile Partitioning - 分块与划分
add_executable(07_tile_partition 07_tile_partition/07_tile_partition.cu)

# 08: WMMA 和 MMA 深入
add_executable(08_wmma_mma 08_wmma_mma/08_wmma_mma.cu)

# 09: TMA - Tensor Memory Accelerator
add_executable(09_tma_copy 09_tma_copy/09_tma_copy.cu)

# 10: Cluster 和 Warp 同步
add_executable(10_cluster_warp 10_cluster_warp/10_cluster_warp.cu)

# ============================================
# 高级教程 (11-14)
# ============================================

# 11: 量化 - INT8/INT4
add_executable(11_quantization 11_quantization/11_quantization.cu)

# 12: 高级 GEMM
add_executable(12_gemm_advanced 12_gemm_advanced/12_gemm_advanced.cu)

# 13: Flash Attention 2
add_executable(13_attention_v2 13_attention_rocm/13_attention_v2.cu)

# 14: 综合性能测试
add_executable(14_benchmark 14_benchmark/14_benchmark.cu)

# 打印成功消息
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  CUTE 专家级教程配置完成!")
message(STATUS "========================================")
message(STATUS "")
message(STATUS "支持 GPU:")
message(STATUS "  - RTX 3060 (SM86)")
message(STATUS "  - A100 (SM80)")
message(STATUS "")
message(STATUS "教程分类:")
message(STATUS "  基础：01-06 (Layout, Swizzle, MMA, GEMM, Attention)")
message(STATUS "  中级：07-10 (Tile, WMMA, TMA, Cluster)")
message(STATUS "  高级：11-14 (量化，高级 GEMM, FlashAttn v2)")
message(STATUS "")
message(STATUS "运行 'make -j8' 编译所有教程")
message(STATUS "========================================")
message(STATUS "")
