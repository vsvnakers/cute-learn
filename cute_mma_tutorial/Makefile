# ============================================================================
# CuTe + MMA 教学 Makefile
# ============================================================================
# 目标架构：SM86 (RTX 3060 Laptop)
# CUDA 版本：12.8
# 依赖：CUTLASS 库 (../cutlass)
# ============================================================================

# ---- 路径配置 ----
CUTLASS_HOME := $(shell cd ../cutlass && pwd)

# ---- 编译器配置 ----
NVCC := nvcc

# ---- 架构配置 (RTX 3060 = SM86) ----
CUDA_ARCH := sm_86
CUDA_ARCH_CODE := -gencode arch=compute_86,code=sm_86

# ---- 编译选项 ----
# --expt-relaxed-constexpr : 允许 host constexpr 在 device 代码中使用
# --expt-extended-lambda   : 允许 extended lambda (__device__ lambda)
# -Xcompiler -fPIC         : 位置无关代码
# -g                       : 调试信息 (用于 ncu 源码级分析)
# -lineinfo                : 行号信息 (用于 ncu 源码映射)
NVCC_FLAGS := -std=c++17 -O3 \
              --expt-relaxed-constexpr \
              --expt-extended-lambda \
              -Xcompiler -fPIC \
              $(CUDA_ARCH_CODE) \
              -g -lineinfo

# ---- 包含路径 ----
NVCC_INCLUDES := -I$(CUTLASS_HOME)/include

# ---- 目标文件列表 ----
# 每个 .cu 文件编译为一个独立的可执行文件
TARGETS := \
    01_layout_basics \
    02_tensor_basics \
    03_copy_basic \
    04_copy_async \
    05_mma_basic \
    06_mma_register_layout \
    07_tiled_mma \
    08_cute_gemm \
    09_async_copy_sm80 \
    10_shared_mem_gemm \
    11_software_pipeline \
    12_perf_async_gemm

# ---- 默认目标：编译所有 ----
all: $(TARGETS)

# ---- 模式规则：.cu -> 可执行文件 ----
%: %.cu
	@echo "================================================"
	@echo "编译: $< -> $@"
	@echo "  架构: $(CUDA_ARCH)"
	@echo "  选项: $(NVCC_FLAGS)"
	@echo "================================================"
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) -o $@ $<
	@echo "完成: $@"
	@echo ""

# ---- 日志目录 ----
LOG_DIR := logs

# ---- 运行所有 (同时保存日志) ----
run: all
	@mkdir -p $(LOG_DIR)
	@for target in $(TARGETS); do \
		echo ""; \
		echo "========== 运行 $$target =========="; \
		./$$target 2>&1 | tee $(LOG_DIR)/$$target.log; \
	done
	@echo ""
	@echo "日志已保存到 $(LOG_DIR)/ 目录"

# ---- 单独运行 (同时保存日志) ----
run_%: %
	@mkdir -p $(LOG_DIR)
	@echo "========== 运行 $* =========="
	./$< 2>&1 | tee $(LOG_DIR)/$*.log

# ---- NCU profiling ----
# ncu 需要 root 权限或 --target-processes all
# 生成 .ncu-rep 文件可用 Nsight Systems GUI 查看
ncu_%: %
	@echo "========== NCU Profiling: $* =========="
	ncu --set full \
	    --kernel-name regex:".*" \
	    --launch-count 1 \
	    --launch-skip 0 \
	    -o $*_ncu \
	    ./$< 2>&1 | tee $*_ncu.txt
	@echo "NCU 报告已保存: $*_ncu.ncu-rep"

# ---- 保存输出到文件 ----
save_output: all
	@for target in $(TARGETS); do \
		echo "保存 $$target 输出..."; \
		./$$target > $${target}_output.txt 2>&1; \
	done
	@echo "所有输出已保存到 *_output.txt"

# ---- 清理 ----
clean:
	rm -f $(TARGETS)
	rm -f *_output.txt *_ncu.txt *_ncu.ncu-rep
	rm -rf $(LOG_DIR)
	rm -f *.o *.d

# ---- 帮助 ----
help:
	@echo "CuTe + MMA 教学 Makefile"
	@echo ""
	@echo "目标文件:"
	@for target in $(TARGETS); do \
		echo "  $$target"; \
	done
	@echo ""
	@echo "常用命令:"
	@echo "  make              - 编译所有教程"
	@echo "  make run          - 编译并运行所有教程"
	@echo "  make run_XXX      - 编译并运行单个教程"
	@echo "  make ncu_XXX      - 对单个教程进行 NCU profiling"
	@echo "  make save_output  - 编译运行并保存输出到文件"
	@echo "  make clean        - 清理编译产物"
	@echo ""
	@echo "配置:"
	@echo "  CUDA_ARCH      = $(CUDA_ARCH)"
	@echo "  CUTLASS_HOME   = $(CUTLASS_HOME)"
	@echo "  NVCC_FLAGS     = $(NVCC_FLAGS)"

.PHONY: all run save_output clean help $(addprefix run_,$(TARGETS)) $(addprefix ncu_,$(TARGETS))
