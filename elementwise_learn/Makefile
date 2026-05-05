CUTLASS_HOME := $(shell cd ../cutlass && pwd)

NVCC := nvcc
NVCC_FLAGS := -std=c++17 -O3 \
              --expt-relaxed-constexpr \
              --expt-extended-lambda \
              -Xcompiler -fPIC \
              -gencode arch=compute_86,code=sm_86

NVCC_INCLUDES := -I$(CUTLASS_HOME)/include

TARGET := elementwise_add

.PHONY: all clean run

all: $(TARGET)

$(TARGET): elementwise_add.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
