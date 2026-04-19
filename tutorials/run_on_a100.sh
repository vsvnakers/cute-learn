#!/bin/bash
# A100 快速编译运行脚本

echo "========================================"
echo "  CUTE 教程 - A100 编译运行脚本"
echo "========================================"

# 检查 GPU
echo ""
echo "检查 GPU..."
nvidia-smi --query-gpu=name,compute_cap --format=csv

# 检查 CUDA
echo ""
echo "检查 CUDA..."
nvcc --version | head -2

# 编译
echo ""
echo "开始编译..."
mkdir -p build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
make -j8

# 运行关键测试
echo ""
echo "========================================"
echo "  运行验证测试"
echo "========================================"

echo ""
echo "[1/4] 运行 05_gemm..."
./05_gemm | grep -E "PASS|FAIL|加速比"

echo ""
echo "[2/4] 运行 11_quantization..."
./11_quantization | grep -E "PASS|FAIL|验证"

echo ""
echo "[3/4] 运行 13_attention_v2..."
./13_attention_v2 | grep -E "结果 | 期望"

echo ""
echo "[4/4] 运行 14_benchmark..."
./14_benchmark | grep -E "GPU|SM 版本 | 加速比"

echo ""
echo "========================================"
echo "  验证完成!"
echo "========================================"
