# Makefile for CUTE Swizzle Learn

# CUDA 编译器
NVCC = nvcc

# 编译选项
CXXFLAGS = -std=c++17
CUDA_FLAGS = -arch=sm_80 -rdc=true

# CUTE 头文件路径（需要根据实际 CUTLASS 安装位置调整）
CUTLASS_PATH = ../cutlass/include
INCLUDES = -I$(CUTLASS_PATH) -I$(CUTLASS_PATH)/cute

# 目标文件
TARGETS = 04_basic 05_advanced 06_practical

# 源文件
SRCS = 04_code_example_basic.cu 05_code_example_advanced.cu 06_code_example_practical.cu

# 默认目标
all: $(TARGETS)

# 编译基础示例
04_basic: 04_code_example_basic.cu
	$(NVCC) $(CXXFLAGS) $(CUDA_FLAGS) $(INCLUDES) $< -o $@

# 编译进阶示例
05_advanced: 05_code_example_advanced.cu
	$(NVCC) $(CXXFLAGS) $(CUDA_FLAGS) $(INCLUDES) $< -o $@

# 编译实战示例
06_practical: 06_code_example_practical.cu
	$(NVCC) $(CXXFLAGS) $(CUDA_FLAGS) $(INCLUDES) $< -o $@

# 运行所有示例
run: $(TARGETS)
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

# 清理
clean:
	rm -f $(TARGETS)

# 单独运行某个示例
run_basic: 04_basic
	./04_basic

run_advanced: 05_advanced
	./05_advanced

run_practical: 06_practical
	./06_practical

# 帮助信息
help:
	@echo "CUTE Swizzle Learn Makefile"
	@echo ""
	@echo "可用目标:"
	@echo "  all          - 编译所有示例 (默认)"
	@echo "  04_basic     - 编译基础示例"
	@echo "  05_advanced  - 编译进阶示例"
	@echo "  06_practical - 编译实战示例"
	@echo "  run          - 编译并运行所有示例"
	@echo "  run_basic    - 运行基础示例"
	@echo "  run_advanced - 运行进阶示例"
	@echo "  run_practical - 运行实战示例"
	@echo "  clean        - 删除编译生成的文件"
	@echo "  help         - 显示此帮助信息"
	@echo ""
	@echo "使用示例:"
	@echo "  make              # 编译所有"
	@echo "  make run_basic    # 运行基础示例"
	@echo "  make clean        # 清理"

.PHONY: all run clean run_basic run_advanced run_practical help
