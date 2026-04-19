#!/bin/bash
# CUTE 教程编译脚本
# 用法：./build.sh [laptop|server|universal]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检测当前 GPU
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
        echo $COMPUTE_CAP
    else
        echo "unknown"
    fi
}

# 清理构建
clean() {
    echo_info "清理构建目录..."
    rm -rf "${BUILD_DIR}"
}

# 编译函数
build() {
    local ARCH=$1
    local DESC=$2

    echo_info "编译配置：$DESC"
    echo_info "目标架构：SM$ARCH"

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    cmake -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" ..
    make -j$(nproc)

    echo ""
    echo_info "========================================"
    echo_info "编译完成！"
    echo_info "========================================"
    echo ""
    echo_info "可执行文件位置：${BUILD_DIR}/"
    echo ""
    echo_info "运行测试："
    echo_info "  ./01_layout_basic     # Layout 基础"
    echo_info "  ./02_swizzle_basic    # Swizzle 技术"
    echo_info "  ./03_bank_conflict    # Bank Conflict"
    echo_info "  ./04_mma_basic        # MMA 指令"
    echo_info "  ./05_gemm             # GEMM 实现"
    echo_info "  ./06_flash_attention  # Flash Attention"
    echo ""
}

# 主程序
case "${1:-auto}" in
    laptop|3060|86)
        build "86" "RTX 3060 笔记本 (SM86)"
        ;;

    server|a100|80)
        build "80" "A100 服务器 (SM80)"
        ;;

    universal|both|all)
        build "80 86" "通用二进制 (RTX 3060 + A100)"
        ;;

    auto)
        GPU=$(detect_gpu)
        echo_info "检测到 GPU 架构：SM${GPU}"

        if [ "$GPU" = "86" ]; then
            echo_info "RTX 3060  detected，编译 SM86 架构..."
            build "86" "RTX 3060 (SM86)"
        elif [ "$GPU" = "80" ]; then
            echo_info "A100 detected，编译 SM80 架构..."
            build "80" "A100 (SM80)"
        elif [ "$GPU" = "87" ]; then
            echo_info "RTX 3080/3090 detected，编译 SM87 架构..."
            build "87" "RTX 3080/3090 (SM87)"
        elif [ "$GPU" = "89" ]; then
            echo_info "RTX 4090 detected，编译 SM89 架构..."
            build "89" "RTX 4090 (SM89)"
        else
            echo_warn "未知 GPU 架构 (SM${GPU})，编译通用二进制..."
            build "80 86" "通用 (SM80 + SM86)"
        fi
        ;;

    clean)
        clean
        echo_info "清理完成！"
        ;;

    *)
        echo "用法：$0 [laptop|server|universal|auto|clean]"
        echo ""
        echo "选项:"
        echo "  laptop    - 仅 RTX 3060 (SM86)"
        echo "  server    - 仅 A100 (SM80)"
        echo "  universal - 两者都支持"
        echo "  auto      - 自动检测当前 GPU (默认)"
        echo "  clean     - 清理构建目录"
        echo ""
        echo "示例:"
        echo "  ./build.sh          # 自动检测"
        echo "  ./build.sh laptop   # 笔记本编译"
        echo "  ./build.sh universal # 通用二进制"
        exit 1
        ;;
esac
