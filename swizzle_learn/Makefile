# Makefile for CUTE Swizzle Learn

# CUDA 编译器
NVCC = nvcc

# 编译选项
CXXFLAGS = -std=c++17

# CUTE 头文件路径
CUTLASS_PATH = ../cutlass/include
INCLUDES = -I$(CUTLASS_PATH) -I$(CUTLASS_PATH)/cute

# ============================================================================
# 常规 Swizzle 示例 (sm_80+)
# ============================================================================

SWIZZLE_ARCH = -arch=sm_80 -rdc=true
SWIZZLE_TARGETS = 04_basic 05_advanced 06_practical

# 默认目标
all: $(SWIZZLE_TARGETS)

04_basic: 04_code_example_basic.cu
	$(NVCC) $(CXXFLAGS) $(SWIZZLE_ARCH) $(INCLUDES) $< -o $@

05_advanced: 05_code_example_advanced.cu
	$(NVCC) $(CXXFLAGS) $(SWIZZLE_ARCH) $(INCLUDES) $< -o $@

06_practical: 06_code_example_practical.cu
	$(NVCC) $(CXXFLAGS) $(SWIZZLE_ARCH) $(INCLUDES) $< -o $@

# ============================================================================
# TMA 示例 (sm_90+, Hopper only)
# ============================================================================

TMA_ARCH = -arch=sm_90 -rdc=true
TMA_TARGETS = 04_tma_basic 05_tma_advanced 06_tma_gemm

tma_all: $(TMA_TARGETS)

04_tma_basic: tma/04_code_tma_basic.cu
	$(NVCC) $(CXXFLAGS) $(TMA_ARCH) $(INCLUDES) $< -o $@

05_tma_advanced: tma/05_code_tma_advanced.cu
	$(NVCC) $(CXXFLAGS) $(TMA_ARCH) $(INCLUDES) $< -o $@

06_tma_gemm: tma/06_code_tma_gemm.cu
	$(NVCC) $(CXXFLAGS) $(TMA_ARCH) $(INCLUDES) $< -o $@

# ============================================================================
# 运行示例
# ============================================================================

run: $(SWIZZLE_TARGETS)
	@echo "========================================"
	@echo "  运行基础示例"
	@echo "========================================"
	./04_basic
	@echo ""
	@echo "========================================"
	@echo "  运行进阶示例"
	@echo "========================================"
	./05_advanced
	@echo ""
	@echo "========================================"
	@echo "  运行实战示例"
	@echo "========================================"
	./06_practical

run_tma: $(TMA_TARGETS)
	@echo "========================================"
	@echo "  运行 TMA 示例"
	@echo "========================================"
	@echo "注意：需要 Hopper (sm_90) GPU!"
	@echo "========================================"
	./04_tma_basic
	@echo ""
	./05_tma_advanced
	@echo ""
	./06_tma_gemm

# ============================================================================
# 清理
# ============================================================================

clean:
	rm -f $(SWIZZLE_TARGETS) $(TMA_TARGETS)

clean_swizzle:
	rm -f $(SWIZZLE_TARGETS)

clean_tma:
	rm -f $(TMA_TARGETS)

# ============================================================================
# 帮助信息
# ============================================================================

help:
	@echo "CUTE Swizzle Learn Makefile"
	@echo ""
	@echo "可用目标:"
	@echo "  常规 Swizzle 示例 (sm_80+):"
	@echo "    all           - 编译所有常规示例 (默认)"
	@echo "    04_basic      - 编译基础示例"
	@echo "    05_advanced   - 编译进阶示例"
	@echo "    06_practical  - 编译实战示例"
	@echo "    run           - 运行所有常规示例"
	@echo ""
	@echo "  TMA 示例 (sm_90+, Hopper only):"
	@echo "    tma_all       - 编译所有 TMA 示例"
	@echo "    04_tma_basic  - 编译 TMA 基础示例"
	@echo "    05_tma_advanced - 编译 TMA 进阶示例"
	@echo "    06_tma_gemm   - 编译 TMA GEMM 示例"
	@echo "    run_tma       - 运行所有 TMA 示例"
	@echo ""
	@echo "  清理:"
	@echo "    clean         - 删除所有编译产物"
	@echo "    clean_swizzle - 删除常规示例编译产物"
	@echo "    clean_tma     - 删除 TMA 示例编译产物"
	@echo ""
	@echo "使用示例:"
	@echo "  make              # 编译所有常规示例"
	@echo "  make tma_all      # 编译所有 TMA 示例"
	@echo "  make run          # 运行常规示例"
	@echo "  make clean        # 清理"
	@echo ""
	@echo "注意：TMA 示例需要 Hopper 架构 GPU (sm_90)"

.PHONY: all tma_all run run_tma clean clean_swizzle clean_tma help
