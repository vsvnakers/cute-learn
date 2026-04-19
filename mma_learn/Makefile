# MMA Makefile

# 架构设置 (默认 SM80，支持 MMA 的最低架构)
CUDA_ARCH ?= sm_80

# 编译器设置
NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -arch=$(CUDA_ARCH)

# 目标文件
BASIC_TARGET = 04_mma_basic
ADVANCED_TARGET = 05_mma_advanced
GEMM_TARGET = 06_mma_gemm

# 源文件
BASIC_SRC = 04_code_mma_basic.cu
ADVANCED_SRC = 05_code_mma_advanced.cu
GEMM_SRC = 06_code_mma_gemm.cu

# 默认目标
all: $(BASIC_TARGET) $(ADVANCED_TARGET) $(GEMM_TARGET)

# 编译基础示例
$(BASIC_TARGET): $(BASIC_SRC)
	@echo "编译基础 MMA 示例..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "完成：$@"

# 编译进阶示例
$(ADVANCED_TARGET): $(ADVANCED_SRC)
	@echo "编译进阶 MMA 示例..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "完成：$@"

# 编译 GEMM 示例
$(GEMM_TARGET): $(GEMM_SRC)
	@echo "编译 MMA GEMM 示例..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "完成：$@"

# 运行所有示例
run: all
	@echo "\n========== 运行基础示例 =========="
	./$(BASIC_TARGET)
	@echo "\n========== 运行进阶示例 =========="
	./$(ADVANCED_TARGET)
	@echo "\n========== 运行 GEMM 示例 =========="
	./$(GEMM_TARGET)

# 保存输出到 txt 文件
save_output: all
	@echo "保存输出到 txt 文件..."
	./$(BASIC_TARGET) > 04_mma_basic.txt 2>&1
	./$(ADVANCED_TARGET) > 05_mma_advanced.txt 2>&1
	./$(GEMM_TARGET) > 06_mma_gemm.txt 2>&1
	@echo "输出已保存到:"
	@echo "  - 04_mma_basic.txt"
	@echo "  - 05_mma_advanced.txt"
	@echo "  - 06_mma_gemm.txt"

# 单独编译目标
04_mma_basic: $(BASIC_TARGET)
05_mma_advanced: $(ADVANCED_TARGET)
06_mma_gemm: $(GEMM_TARGET)

# 清理
clean:
	rm -f $(BASIC_TARGET) $(ADVANCED_TARGET) $(GEMM_TARGET)
	rm -f *.txt *.o *.d

# 帮助
help:
	@echo "MMA 学习 Makefile 使用说明:"
	@echo "  make          - 编译所有示例"
	@echo "  make 04_mma_basic    - 编译基础示例"
	@echo "  make 05_mma_advanced - 编译进阶示例"
	@echo "  make 06_mma_gemm     - 编译 GEMM 示例"
	@echo "  make run      - 编译并运行所有示例"
	@echo "  make save_output     - 编译运行并保存输出到 txt"
	@echo "  make clean    - 清理编译产物"
	@echo ""
	@echo "环境变量:"
	@echo "  CUDA_ARCH   - CUDA 架构 (默认：sm_80)"

.PHONY: all run save_output clean help 04_mma_basic 05_mma_advanced 06_mma_gemm
