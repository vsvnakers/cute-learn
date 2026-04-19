# 第三课：PTX 指令集详解

## 3.1 PTX 指令分类

### 完整指令分类

```
┌─────────────────────────────────────────────────────────────────┐
│  PTX 指令集分类                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 数据移动指令 (Data Movement)                                │
│     ld, st, mov, cvt, mergeo, mergelo, ...                     │
│                                                                 │
│  2. 算术指令 (Arithmetic)                                       │
│     add, sub, mul, div, fma, neg, abs, ...                     │
│                                                                 │
│  3. 比较指令 (Comparison)                                       │
│     setp, set, min, max, ...                                   │
│                                                                 │
│  4. 控制流指令 (Control Flow)                                   │
│     bra, call, ret, brk, cont, ...                             │
│                                                                 │
│  5. 并行同步指令 (Parallel Synchronization)                     │
│     bar, atom, red, membar, ...                                │
│                                                                 │
│  6. 特殊指令 (Special)                                          │
│     sqrt, sin, cos, exp, log, ...                              │
│                                                                 │
│  7. 纹理内存指令 (Texture Memory)                               │
│     tex, tld4, ...                                             │
│                                                                 │
│  8. 表面内存指令 (Surface Memory)                               │
│     surf, ...                                                  │
│                                                                 │
│  9. Tensor Core 指令 (MMA)                                      │
│     mma.sync, ...                                              │
│                                                                 │
│  10. TMA 指令 (Tensor Memory Accelerator, SM90+)                │
│     tensormap.cp, mbarrier, ...                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3.2 数据移动指令

### 加载指令 (ld)

```ptx
// 语法：ld.<space>.<type> <dest>, [<src>];

// 从全局内存加载
ld.global.f32 %f1, [%rd1];          // 加载 32 位浮点
ld.global.u32 %r1, [%rd1];          // 加载 32 位无符号整数
ld.global.u64 %rd1, [%rd2];         // 加载 64 位

// 从共享内存加载
ld.shared.f32 %f1, [%rd1];          // 加载共享内存
ld.shared.v4.f32 {%v4f}, [%rd1];    // 向量加载 4×f32

// 从常量内存加载
ld.const.f32 %f1, [%rd1];           // 加载常量内存

// 带缓存提示的加载
ld.global.cg.f32 %f1, [%rd1];       // 流式加载（缓存全局）
ld.global.cs.f32 %f1, [%rd1];       // 流式加载（不缓存）
ld.global.lu.f32 %f1, [%rd1];       // 最后使用提示
ld.global.cv.f32 %f1, [%rd1];       // 计算向量（缓存行）

// 带对齐提示的加载
ld.global.aligned.f32 %f1, [%rd1];  // 对齐加载（性能更好）
```

### 存储指令 (st)

```ptx
// 语法：st.<space>.<type> [<dst>], <src>;

// 存储到全局内存
st.global.f32 [%rd1], %f1;          // 存储 32 位浮点
st.global.u32 [%rd1], %r1;          // 存储 32 位无符号整数

// 存储到共享内存
st.shared.f32 [%rd1], %f1;          // 存储共享内存
st.shared.v4.f32 [%rd1], {%v4f};    // 向量存储

// 带缓存提示的存储
st.global.cg.f32 [%rd1], %f1;       // 缓存全局
st.global.cs.f32 [%rd1], %f1;       // 流式存储（不缓存）
st.global.wt.f32 [%rd1], %f1;       // 写透（write-through）

// 带对齐提示的存储
st.global.aligned.f32 [%rd1], %f1;  // 对齐存储
```

### 移动指令 (mov)

```ptx
// 语法：mov.<type> <dest>, <src>;

// 基本移动
mov.b32 %r1, %r2;                   // 32 位移动
mov.b64 %rd1, %rd2;                 // 64 位移动
mov.f32 %f1, %f2;                   // 浮点移动

// 类型重解释（位模式不变）
mov.b32 %f1, %r1;                   // u32 → f32
mov.b32 %r1, %f1;                   // f32 → u32

// 立即数移动
mov.u32 %r1, 42;                    // 移动立即数
mov.b64 %rd1, 0x123456789ABCDEF0;   // 64 位立即数

// 特殊寄存器移动
mov.u32 %r1, %tid.x;                // 读取线程索引
mov.u32 %r1, %ctaid.x;              // 读取块索引
mov.u64 %rd1, %clock64;             // 读取时钟
```

### 转换指令 (cvt)

```ptx
// 语法：cvt.<mode>.<dst_type>.<src_type> <dest>, <src>;

// 整数转换
cvt.u16.u8 %r1, %r2;                // u8 → u16
cvt.u32.u16 %r1, %r2;               // u16 → u32
cvt.u8.u32 %r1, %r2;                // u32 → u8 (截断)

// 浮点转换
cvt.f32.f64 %f1, %fd1;              // f64 → f32
cvt.f64.f32 %fd1, %f1;              // f32 → f64

// 浮点←→整数
cvt.u32.f32 %r1, %f1;               // f32 → u32
cvt.f32.u32 %f1, %r1;               // u32 → f32

// 带舍入模式的转换
cvt.rn.f16.f32 %h1, %f1;            // f32→f16 (最近舍入)
cvt.rz.f16.f32 %h1, %f1;            // f32→f16 (向零舍入)

// 带饱和的转换
cvt.sat.u32.f32 %r1, %f1;           // f32→u32 (饱和)
```

## 3.3 算术指令

### 加法指令 (add)

```ptx
// 语法：add.<type> <dest>, <src1>, <src2>;

// 整数加法
add.u32 %r1, %r2, %r3;              // u32 加法
add.s32 %r1, %r2, %r3;              // s32 加法
add.u64 %rd1, %rd2, %rd3;           // u64 加法

// 浮点加法
add.f32 %f1, %f2, %f3;              // f32 加法
add.f64 %fd1, %fd2, %fd3;           // f64 加法

// 带修饰的加法
add.sat.u32 %r1, %r2, %r3;          // 饱和加法（溢出时饱和）
add.ftz.f32 %f1, %f2, %f3;          // FTZ 模式（非规格化数视为零）
```

### 减法指令 (sub)

```ptx
// 语法：sub.<type> <dest>, <src1>, <src2>;

// 整数减法
sub.u32 %r1, %r2, %r3;              // u32 减法
sub.s32 %r1, %r2, %r3;              // s32 减法

// 浮点减法
sub.f32 %f1, %f2, %f3;              // f32 减法
sub.f64 %fd1, %fd2, %fd3;           // f64 减法
```

### 乘法指令 (mul)

```ptx
// 语法：mul.<mode>.<type> <dest>, <src1>, <src2>;

// 整数乘法
mul.lo.u32 %r1, %r2, %r3;           // u32 乘法（取低位 32 位）
mul.hi.u32 %r1, %r2, %r3;           // u32 乘法（取高位 32 位）
mul.lo.s32 %r1, %r2, %r3;           // s32 乘法（低位）
mul.hi.s32 %r1, %r2, %r3;           // s32 乘法（高位）

// 64 位乘法结果
mul.wide.u32 %rd1, %r2, %r3;        // u32×u32→u64

// 浮点乘法
mul.f32 %f1, %f2, %f3;              // f32 乘法
mul.f64 %fd1, %fd2, %fd3;           // f64 乘法

// 浮点乘法修饰
mul.ftz.f32 %f1, %f2, %f3;          // FTZ 模式
mul.rn.f32 %f1, %f2, %f3;           // 指定舍入模式
```

### 乘加指令 (fma)

```ptx
// 语法：fma.<rnd>.<type> <dest>, <src1>, <src2>, <src3>;
// 计算：dest = src1 × src2 + src3

// 浮点乘加
fma.rn.f32 %f1, %f2, %f3, %f4;      // f32 乘加（最近舍入）
fma.rz.f32 %f1, %f2, %f3, %f4;      // f32 乘加（向零舍入）
fma.rn.f64 %fd1, %fd2, %fd3, %fd4;  // f64 乘加

// FTZ 模式
fma.ftz.f32 %f1, %f2, %f3, %f4;     // FTZ 乘加

// 示例：点积计算
// result = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
fma.rn.f32 %f1, %a0, %b0, %a1;      // f1 = a0*b0 + a1
fma.rn.f32 %f2, %a2, %b2, %a3;      // f2 = a2*b2 + a3
fma.rn.f32 %result, %b1, %f1, %f2;  // result = b1*f1 + f2
```

### 除法指令 (div)

```ptx
// 语法：div.<type> <dest>, <src1>, <src2>;

// 整数除法
div.u32 %r1, %r2, %r3;              // u32 除法
div.s32 %r1, %r2, %r3;              // s32 除法
rem.u32 %r1, %r2, %r3;              // u32 取余
rem.s32 %r1, %r2, %r3;              // s32 取余

// 浮点除法
div.f32 %f1, %f2, %f3;              // f32 除法
div.f64 %fd1, %fd2, %fd3;           // f64 除法

// 近似倒数（更快）
rcp.approx.f32 %f1, %f2;            // 近似倒数 1/x
rcp.approx.ftz.f32 %f1, %f2;        // 近似倒数 + FTZ

// 使用倒数实现除法
rcp.approx.f32 %f_inv, %f3;         // f_inv = 1/f3
mul.f32 %f1, %f2, %f_inv;           // f1 = f2 * (1/f3)
```

### 绝对值和取反

```ptx
// 绝对值
abs.s32 %r1, %r2;                   // s32 绝对值
abs.f32 %f1, %f2;                   // f32 绝对值

// 取反
neg.s32 %r1, %r2;                   // s32 取反
neg.f32 %f1, %f2;                   // f32 取反
```

## 3.4 比较指令

### 谓词设置指令 (setp)

```ptx
// 语法：setp.<cond>.<type> <dest_pred>, <src1>, <src2>;

// 整数比较
setp.lt.u32 %p1, %r1, %r2;          // %p1 = (%r1 < %r2)
setp.gt.u32 %p1, %r1, %r2;          // %p1 = (%r1 > %r2)
setp.eq.u32 %p1, %r1, %r2;          // %p1 = (%r1 == %r2)
setp.le.u32 %p1, %r1, %r2;          // %p1 = (%r1 <= %r2)
setp.ge.u32 %p1, %r1, %r2;          // %p1 = (%r1 >= %r2)
setp.ne.u32 %p1, %r1, %r2;          // %p1 = (%r1 != %r2)

// 有符号整数比较
setp.lt.s32 %p1, %r1, %r2;          // %p1 = (%r1 < %r2) 有符号

// 浮点比较
setp.lt.f32 %p1, %f1, %f2;          // %p1 = (%f1 < %f2)
setp.eq.f32 %p1, %f1, %f2;          // %p1 = (%f1 == %f2)

// 无序比较（处理 NaN）
setp.ltu.f32 %p1, %f1, %f2;         // %p1 = (%f1 < %f2) 或无序
setp.equ.f32 %p1, %f1, %f2;         // %p1 = (%f1 == %f2) 或无序

// 特殊值检查
setp.nan.f32 %p1, %f1;              // %p1 = isnan(%f1)
setp.inf.f32 %p1, %f1;              // %p1 = isinf(%f1)
```

### 寄存器设置指令 (set)

```ptx
// 语法：set.<cond>.<type> <dest>, <src1>, <src2>;

// 比较并设置寄存器（结果 0 或 1）
set.lt.u32 %r1, %r2, %r3;           // %r1 = (%r2 < %r3) ? 1 : 0
set.eq.u32 %r1, %r2, %r3;           // %r1 = (%r2 == %r3) ? 1 : 0

// 与立即数比较
set.eq.u32 %r1, %r2, 0;             // %r1 = (%r2 == 0) ? 1 : 0
```

### 最小/最大值指令

```ptx
// 语法：min/max.<type> <dest>, <src1>, <src2>;

// 整数最小/最大
min.u32 %r1, %r2, %r3;              // %r1 = min(%r2, %r3) 无符号
min.s32 %r1, %r2, %r3;              // %r1 = min(%r2, %r3) 有符号
max.u32 %r1, %r2, %r3;              // %r1 = max(%r2, %r3)

// 浮点最小/最大
min.f32 %f1, %f2, %f3;              // %f1 = min(%f2, %f3)
max.f32 %f1, %f2, %f3;              // %f1 = max(%f2, %f3)

// 带 FTZ 的浮点最小/最大
min.ftz.f32 %f1, %f2, %f3;          // FTZ 模式
```

## 3.5 位操作指令

### 逻辑位操作

```ptx
// 语法：and/or/xor/not.<type> <dest>, <src1>, <src2>;

// 按位与
and.b32 %r1, %r2, %r3;              // %r1 = %r2 & %r3
and.b64 %rd1, %rd2, %rd3;           // %rd1 = %rd2 & %rd3

// 按位或
or.b32 %r1, %r2, %r3;               // %r1 = %r2 | %r3

// 按位异或
xor.b32 %r1, %r2, %r3;              // %r1 = %r2 ^ %r3

// 按位取反
not.b32 %r1, %r2;                   // %r1 = ~%r2
```

### 位移操作

```ptx
// 语法：shl/shr/ashr.<type> <dest>, <src>, <shift>;

// 左移
shl.b32 %r1, %r2, %r3;              // %r1 = %r2 << %r3
shl.b64 %rd1, %rd2, %r3;            // %rd1 = %rd2 << %r3

// 逻辑右移（无符号）
shr.b32 %r1, %r2, %r3;              // %r1 = %r2 >> %r3 (零填充)

// 算术右移（有符号）
ashr.b32 %r1, %r2, %r3;             // %r1 = %r2 >> %r3 (符号填充)

// 循环移位
shf.l.wrap.b32 %r1, %r2, %r3, %r4;  // 循环左移
shf.r.wrap.b32 %r1, %r2, %r3, %r4;  // 循环右移
```

### 位查找指令

```ptx
// 前导零计数
clz.u32 %r1, %r2;                   // 计算高位的零个数

// 找第一个为 1 的位
ffs.u32 %r1, %r2;                   // 从 1 开始计数
bfind.u32 %r1, %r2;                 // 从 0 开始计数

// 1 的个数计数
popc.u32 %r1, %r2;                  // 计算 1 的个数

// 字节交换
brev.b32 %r1, %r2;                  // 位反转
```

## 3.6 控制流指令

### 无条件分支

```ptx
// 语法：bra <label>;

bra LOOP_START;                     // 跳转到 LOOP_START
bra EXIT;                           // 跳转到 EXIT
```

### 条件分支

```ptx
// 语法：@<pred> bra <label>;

@%p1 bra EXIT;                      // 如果 %p1 为真，跳转到 EXIT
@!%p1 bra LOOP_START;               // 如果 %p1 为假，跳转到 LOOP_START

// 示例：if-else 结构
setp.lt.u32 %p1, %r1, %r2;
@%p1 bra THEN_BLOCK;
// ELSE 部分
bra END_IF;
THEN_BLOCK:
// THEN 部分
END_IF:
```

### 函数调用和返回

```ptx
// 函数定义
.func (.param .u32 result) square_func(.param .u32 input) {
    ld.param.u32 %r1, [input];
    mul.u32 %r2, %r1, %r1;
    st.param.u32 [result], %r2;
    ret;
}

// 函数调用
call square_func(%r_out, %r_in);

// 返回
ret;                                // 从函数返回
```

### 循环结构

```ptx
// while 循环
LOOP_BEGIN:
setp.lt.u32 %p1, %r1, %r2;
@!%p1 bra LOOP_END;
// 循环体
add.u32 %r1, %r1, 1;
bra LOOP_BEGIN;
LOOP_END:

// for 循环 (i = 0; i < n; i++)
mov.u32 %r1, 0;                     // i = 0
FOR_LOOP:
setp.ge.u32 %p1, %r1, %r2;          // i >= n?
@%p1 bra FOR_END;
// 循环体
add.u32 %r1, %r1, 1;                // i++
bra FOR_LOOP;
FOR_END:
```

## 3.7 并行同步指令

### 屏障同步

```ptx
// 块内屏障
bar.sync 0;                         // 块内所有线程同步

// 带计数器的屏障
bar.arrive 0;                       // 到达屏障
bar.wait 0;                         // 等待屏障

// 共享内存屏障
membar.cta;                         // CTA 内内存屏障
membar.gl;                          // 全局内存屏障
membar.sys;                         // 系统级内存屏障
```

### 原子操作

```ptx
// 原子加法
atom.add.u32 %r1, [%rd1], %r2;      // *%rd1 += %r2, 返回旧值

// 原子减法
atom.sub.u32 %r1, [%rd1], %r2;      // *%rd1 -= %r2

// 原子交换
atom.exch.u32 %r1, [%rd1], %r2;     // 交换 *%rd1 和 %r2

// 原子比较交换
atom.cas.u32 %r1, [%rd1], %r2, %r3; // if (*%rd1 == %r2) *%rd1 = %r3

// 原子最小/最大
atom.min.u32 %r1, [%rd1], %r2;      // *%rd1 = min(*%rd1, %r2)
atom.max.u32 %r1, [%rd1], %r2;      // *%rd1 = max(*%rd1, %r2)

// 原子与/或/异或
atom.and.u32 %r1, [%rd1], %r2;      // *%rd1 &= %r2
atom.or.u32 %r1, [%rd1], %r2;       // *%rd1 |= %r2
atom.xor.u32 %r1, [%rd1], %r2;      // *%rd1 ^= %r2
```

### 归约操作

```ptx
// 共享内存归约
red.shared.add.u32 [%rd1], %r2;     // 共享内存原子加
red.shared.min.u32 [%rd1], %r2;     // 共享内存原子最小
red.shared.max.u32 [%rd1], %r2;     // 共享内存原子最大
```

## 3.8 特殊功能指令

### 平方根

```ptx
// 精确平方根
sqrt.rn.f32 %f1, %f2;               // %f1 = sqrt(%f2)
sqrt.rn.f64 %fd1, %fd2;             // %fd1 = sqrt(%fd2)

// 近似平方根（更快）
sqrt.approx.f32 %f1, %f2;           // 近似平方根
sqrt.approx.ftz.f32 %f1, %f2;       // 近似平方根 + FTZ
```

### 三角函数

```ptx
// 正弦
sin.approx.f32 %f1, %f2;            // 近似正弦
sin.rn.f32 %f1, %f2;                // 精确正弦

// 余弦
cos.approx.f32 %f1, %f2;            // 近似余弦

// 正切
tan.approx.f32 %f1, %f2;            // 近似正切
```

### 指数和对数

```ptx
// 自然指数
exp.approx.f32 %f1, %f2;            // 近似 e^x
exp.rn.f32 %f1, %f2;                // 精确 e^x

// 自然对数
log.approx.f32 %f1, %f2;            // 近似 ln(x)
log.rn.f32 %f1, %f2;                // 精确 ln(x)

// 以 2 为底的对数
log2.approx.f32 %f1, %f2;           // 近似 log2(x)

// 以 10 为底的对数
log10.approx.f32 %f1, %f2;          // 近似 log10(x)
```

## 3.9 内联 PTX 示例

### C++ 内联 PTX 语法

```cpp
// 基本语法
asm volatile("PTX 指令"
             : 输出操作数
             : 输入操作数
             : 可选的 clobber 列表);

// 操作数约束:
// =r, =f  输出寄存器
// r, f    输入寄存器
// l       64 位指针
// h       16 位指针
```

### 内联 PTX 示例

```cpp
// 示例 1: 快速平方根
__device__ float fast_sqrt(float x) {
    float result;
    asm("sqrt.approx.ftz.f32 %0, %1;"
        : "=f"(result)
        : "f"(x));
    return result;
}

// 示例 2: 原子加法
__device__ unsigned int atomic_add(unsigned int* addr, unsigned int val) {
    unsigned int ret;
    asm("atom.add.u32 %0, [%1], %2;"
        : "=r"(ret)
        : "l"(addr), "r"(val));
    return ret;
}

// 示例 3: 乘加操作
__device__ float fma_fast(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(result)
        : "f"(a), "f"(b), "f"(c));
    return result;
}

// 示例 4: 线程索引读取
__device__ int get_thread_id() {
    int tid;
    asm("mov.u32 %0, %tid.x;"
        : "=r"(tid));
    return tid;
}

// 示例 5: 时钟周期读取
__device__ unsigned long long get_clock() {
    unsigned long long clock;
    asm("mov.u64 %0, %clock64;"
        : "=l"(clock));
    return clock;
}
```

## 3.10 小结

| 指令类别 | 代表指令 | 用途 |
|---------|---------|------|
| 数据移动 | ld, st, mov, cvt | 内存访问、类型转换 |
| 算术运算 | add, sub, mul, fma | 基本计算 |
| 比较 | setp, set, min, max | 条件判断 |
| 控制流 | bra, call, ret | 分支、循环、函数 |
| 同步 | bar, atom, membar | 线程同步 |
| 特殊功能 | sqrt, sin, exp, log | 数学函数 |

**下一步**: 阅读 [04_code_ptx_basic.cu](04_code_ptx_basic.cu) 开始实践
