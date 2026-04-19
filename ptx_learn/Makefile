# PTX Makefile

# 架构设置
CUDA_ARCH ?= sm_80

# 编译器设置
NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -arch=$(CUDA_ARCH)

# 目标文件
BASIC_TARGET = 04_ptx_basic
ADVANCED_TARGET = 05_ptx_advanced
INTRINSICS_TARGET = 06_ptx_intrinsics

# 源文件
BASIC_SRC = 04_code_ptx_basic.cu
ADVANCED_SRC = 05_code_ptx_advanced.cu
INTRINSICS_SRC = 06_code_ptx_intrinsics.cu

# 默认目标
all: $(BASIC_TARGET) $(ADVANCED_TARGET) $(INTRINSICS_TARGET)

# 编译基础示例
$(BASIC_TARGET): $(BASIC_SRC)
	@echo "编译基础 PTX 示例..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "完成：$@"

# 编译进阶示例
$(ADVANCED_TARGET): $(ADVANCED_SRC)
	@echo "编译进阶 PTX 示例..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "完成：$@"

# 编译内联汇编示例
$(INTRINSICS_TARGET): $(INTRINSICS_SRC)
	@echo "编译 PTX 内联汇编示例..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "完成：$@"

# 运行所有示例
run: all
	@echo "\n========== 运行基础示例 =========="
	./$(BASIC_TARGET)
	@echo "\n========== 运行进阶示例 =========="
	./$(ADVANCED_TARGET)
	@echo "\n========== 运行内联汇编示例 =========="
	./$(INTRINSICS_TARGET)

# 保存输出到 txt 文件
save_output: all
	@echo "保存输出到 txt 文件..."
	./$(BASIC_TARGET) > 04_ptx_basic.txt 2>&1
	./$(ADVANCED_TARGET) > 05_ptx_advanced.txt 2>&1
	./$(INTRINSICS_TARGET) > 06_ptx_intrinsics.txt 2>&1
	@echo "输出已保存到:"
	@echo "  - 04_ptx_basic.txt"
	@echo "  - 05_ptx_advanced.txt"
	@echo "  - 06_ptx_intrinsics.txt"

# 单独编译目标
04_ptx_basic: $(BASIC_TARGET)
05_ptx_advanced: $(ADVANCED_TARGET)
06_ptx_intrinsics: $(INTRINSICS_TARGET)

# 生成 PTX 文件
ptx: $(BASIC_SRC) $(ADVANCED_SRC) $(INTRINSICS_SRC)
	@echo "生成 PTX 文件..."
	$(NVCC) -ptx $(NVCC_FLAGS) -o 04_ptx_basic.ptx $(BASIC_SRC)
	$(NVCC) -ptx $(NVCC_FLAGS) -o 05_ptx_advanced.ptx $(ADVANCED_SRC)
	$(NVCC) -ptx $(NVCC_FLAGS) -o 06_ptx_intrinsics.ptx $(INTRINSICS_SRC)
	@echo "PTX 文件已生成"

# 查看 PTX 代码
view_ptx: ptx
	@echo "查看 PTX 代码..."
	cat 04_ptx_basic.ptx

# 清理
clean:
	rm -f $(BASIC_TARGET) $(ADVANCED_TARGET) $(INTRINSICS_TARGET)
	rm -f *.txt *.o *.d *.ptx *.cubin

# 帮助
help:
	@echo "PTX 学习 Makefile 使用说明:"
	@echo "  make          - 编译所有示例"
	@echo "  make 04_ptx_basic    - 编译基础示例"
	@echo "  make 05_ptx_advanced - 编译进阶示例"
	@echo "  make 06_ptx_intrinsics - 编译内联汇编示例"
	@echo "  make run      - 编译并运行所有示例"
	@echo "  make save_output     - 编译运行并保存输出到 txt"
	@echo "  make ptx      - 生成 PTX 文件"
	@echo "  make view_ptx - 查看 PTX 代码"
	@echo "  make clean    - 清理编译产物"
	@echo ""
	@echo "环境变量:"
	@echo "  CUDA_ARCH   - CUDA 架构 (默认：sm_80)"

.PHONY: all run save_output clean help ptx view_ptx 04_ptx_basic 05_ptx_advanced 06_ptx_intrinsics
