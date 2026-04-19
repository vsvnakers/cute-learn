# 第七课：PTX 调试技巧

## 7.1 查看 PTX 代码

### 使用 nvcc 生成 PTX

```bash
# 方法 1: 直接生成 PTX 文件
nvcc -ptx my_kernel.cu -o my_kernel.ptx

# 方法 2: 指定架构
nvcc -ptx -arch=sm_80 my_kernel.cu -o my_kernel.ptx

# 方法 3: 生成多个架构的 PTX
nvcc -ptx -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_90,code=sm_90 \
       my_kernel.cu -o my_kernel.ptx

# 方法 4: 保留中间文件
nvcc -keep my_kernel.cu
# 生成 my_kernel.cpp1.ii, my_kernel.ptx, my_kernel.cubin
```

### 查看 PTX 内容

```bash
# 查看 PTX 文件
cat my_kernel.ptx

# 搜索特定 kernel
grep -A 50 ".entry vector_add" my_kernel.ptx

# 查看 PTX 版本和目标
head -10 my_kernel.ptx
```

## 7.2 查看 SASS 代码

### 使用 cuobjdump

```bash
# 从可执行文件中提取 SASS
cuobjdump --dump-sass my_kernel

# 提取特定架构的 SASS
cuobjdump --dump-sass -arch sm_80 my_kernel

# 提取 PTX
cuobjdump --dump-ptx my_kernel

# 提取所有信息
cuobjdump --dump-all my_kernel
```

### 使用 nvdisasm

```bash
# 反编译 cubin 文件
nvdisasm my_kernel.cubin

# 生成可读的 SASS
nvdisasm -gi my_kernel.cubin

# 只查看特定 kernel
nvdisasm -gi -entry vector_add my_kernel.cubin
```

### SASS 代码示例

```
# SASS 代码示例 (Ampere SM80)

vector_add(float*, float*, float*, int):
/*0000*/                   mov.u32 %r1, %ctaid.x;
/*0008*/                   mov.u32 %r2, %ntid.x;
/*0010*/                   mul.lo.u32 %r2, %r1, %r2;
/*0018*/                   mov.u32 %r1, %tid.x;
/*0020*/                   add.u32 %r1, %r2, %r1;
/*0028*/                   setp.ge.u32 %p1, %r1, %rd4;
/*0030*/                   @%p1 bra done;
/*0038*/                   shl.u64 %rd5, %rd5, 2;
/*0040*/                   ld.global.f32 %f1, [%rd5];
/*0048*/                   add.f32 %f1, %f1, %f2;
/*0050*/                   st.global.f32 [%rd6], %f1;
done:
/*0058*/                   ret;
```

## 7.3 使用 Nsight Compute

### 启动分析

```bash
# 基本分析
ncu ./my_kernel

# 详细分析
ncu --set full ./my_kernel

# 指定 kernel
ncu --kernel-name vector_add ./my_kernel

# 导出报告
ncu --set full --export report ./my_kernel
```

### 查看 PTX 和 SASS 关联

```bash
# 查看 PTX 到 SASS 的映射
ncu --dump-config --dump-ptx-sass-mapping ./my_kernel

# 生成 JSON 报告
ncu --json --export report.json ./my_kernel
```

## 7.4 PTX 调试技术

### 添加调试输出

```cpp
__global__ void debug_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 方法 1: printf 调试
    if (idx < 4) {
        printf("线程 %d: data[%d] = %.2f\n", idx, idx, data[idx]);
    }

    // 方法 2: 使用 PTX 读取寄存器
    unsigned int tid;
    asm volatile("mov.u32 %0, %tid.x;" : "=r"(tid));

    unsigned int bid;
    asm volatile("mov.u32 %0, %ctaid.x;" : "=r"(bid));

    if (idx == 0) {
        printf("TID=%u, BID=%u\n", tid, bid);
    }

    // 方法 3: 时钟调试
    unsigned long long start = 0;
    asm volatile("mov.u64 %0, %clock64;" : "=l"(start));

    // ... 计算代码 ...

    unsigned long long end = 0;
    asm volatile("mov.u64 %0, %clock64;" : "=l"(end));

    if (idx == 0) {
        printf("耗时：%llu 周期\n", end - start);
    }
}
```

### 使用 CUDA GDB

```bash
# 启动 CUDA GDB
cuda-gdb ./my_kernel

# 常用命令
(cuda-gdb) break vector_add      # 设置断点
(cuda-gdb) run                   # 运行程序
(cuda-gdb) stepi                 # 单步执行
(cuda-gdb) info registers        # 查看寄存器
(cuda-gdb) print %r1             # 打印寄存器值
(cuda-gdb) continue              # 继续执行
```

## 7.5 性能分析技术

### 使用 nvprof (旧版)

```bash
# 基本性能分析
nvprof ./my_kernel

# 查看详细事件
nvprof --events all ./my_kernel

# 分析内存使用
nvprof --metrics all ./my_kernel

# 导出到文件
nvprof -o profile.nvvp ./my_kernel
```

### 使用 Nsight Systems

```bash
# 启动分析
nsys profile ./my_kernel

# 生成报告
nsys stats report.qdrep

# 查看时间线
nsys ui report.qdrep
```

## 7.6 常见 PTX 错误

### 错误 1: 寄存器类型不匹配

```ptx
// 错误：类型不匹配
mov.f32 %r1, %f1;    // %r1 是整数寄存器

// 正确：使用 mov.b32
mov.b32 %r1, %f1;    // 位模式重解释
```

### 错误 2: 内存空间不匹配

```ptx
// 错误：使用错误的内存空间
ld.global.f32 %f1, [%rd1];  // 但 %rd1 指向 shared 内存

// 正确：使用匹配的内存空间
ld.shared.f32 %f1, [%rd1];  // shared 内存
```

### 错误 3: 对齐问题

```ptx
// 可能导致性能下降
ld.global.f32 %f1, [%rd1];  // %rd1 未对齐

// 优化：确保对齐
ld.global.aligned.f32 %f1, [%rd1];  // 对齐加载
```

### 错误 4: 内联 PTX 约束错误

```cpp
// 错误：约束不匹配
asm volatile("add.f32 %0, %1, %2;"
             : "=r"(result)  // 错误：应该是"f"
             : "f"(a), "f"(b));

// 正确
asm volatile("add.f32 %0, %1, %2;"
             : "=f"(result)
             : "f"(a), "f"(b));
```

## 7.7 优化技巧

### 技巧 1: 使用 FMA 代替 MUL+ADD

```ptx
// 低效：两条指令
mul.f32 %f1, %f2, %f3;
add.f32 %f4, %f1, %f5;

// 高效：一条 FMA 指令
fma.rn.f32 %f4, %f2, %f3, %f5;
```

### 技巧 2: 使用近似指令

```ptx
// 精确但慢
sqrt.rn.f32 %f1, %f2;
rcp.rn.f32 %f1, %f2;

// 快速近似
sqrt.approx.ftz.f32 %f1, %f2;
rcp.approx.ftz.f32 %f1, %f2;
```

### 技巧 3: 使用向量加载/存储

```ptx
// 低效：4 次独立加载
ld.global.f32 %f1, [%rd1];
ld.global.f32 %f2, [%rd1+4];
ld.global.f32 %f3, [%rd1+8];
ld.global.f32 %f4, [%rd1+12];

// 高效：1 次向量加载
ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];
```

### 技巧 4: 使用谓词避免分支

```ptx
// 低效：分支
setp.gt.f32 %p1, %f1, 0.0f;
@%p1 bra positive;
neg.f32 %f2, %f1;
bra done;
positive:
mov.f32 %f2, %f1;
done:

// 高效：谓词执行
setp.gt.f32 %p1, %f1, 0.0f;
abs.f32 %f2, %f1;  // 直接计算，无需分支
```

## 7.8 调试工具总结

| 工具 | 用途 | 命令 |
|------|------|------|
| nvcc -ptx | 生成 PTX | `nvcc -ptx file.cu` |
| cuobjdump | 提取 SASS/PTX | `cuobjdump --dump-sass` |
| nvdisasm | 反编译 cubin | `nvdisasm -gi file.cubin` |
| Nsight Compute | 性能分析 | `ncu ./program` |
| Nsight Systems | 系统级分析 | `nsys profile ./program` |
| CUDA GDB | 调试器 | `cuda-gdb ./program` |
| nvprof | 性能分析 (旧) | `nvprof ./program` |

## 7.9 小结

**调试流程**:
1. 使用 `nvcc -ptx` 生成 PTX 代码
2. 使用 `cuobjdump` 或 `nvdisasm` 查看 SASS
3. 使用 Nsight Compute 分析性能瓶颈
4. 使用 printf 或 CUDA GDB 调试逻辑错误
5. 根据分析结果优化 PTX 代码

**下一步**: 阅读 [08_PTX 学习总结.md](08_PTX 学习总结.md)
