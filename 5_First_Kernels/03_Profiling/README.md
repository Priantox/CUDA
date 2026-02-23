# CUDA Profiling and Nsight – Complete Guide

## Table of Contents

1. [Introduction](#introduction)
2. [What is CUDA Profiling?](#what-is-cuda-profiling)
3. [Why Profiling Matters](#why-profiling-matters)
4. [NVTX – Annotating Your Code](#nvtx--annotating-your-code)
5. [Nsight Tools Overview](#nsight-tools-overview)
6. [Nsight Systems – System-Wide Analysis](#nsight-systems--system-wide-analysis)
7. [Nsight Compute – Deep Kernel Analysis](#nsight-compute--deep-kernel-analysis)
8. [Nsight Systems vs Nsight Compute](#nsight-systems-vs-nsight-compute)
9. [Step-by-Step Profiling Workflow](#step-by-step-profiling-workflow)
10. [Understanding Profiling Output](#understanding-profiling-output)
11. [Common Performance Bottlenecks](#common-performance-bottlenecks)
12. [Key Takeaways](#key-takeaways)

---

## Introduction

Writing a CUDA program that produces correct results is only half the job.  
The other half is making it run **fast and efficiently** on the GPU.

A GPU has thousands of cores and high-bandwidth memory — but if your program is poorly structured, most of that hardware sits idle while your program crawls along.

**CUDA Profiling** is the process of measuring exactly what your program is doing, when, and for how long. It answers critical questions such as:

- Which part of my program is the slowest?
- Is the GPU actually busy, or is it waiting on memory?
- How much time is wasted copying data between CPU and GPU?
- Are my CUDA kernels using the hardware efficiently?

Without profiling, optimizing a GPU program is essentially **guesswork**. With profiling, you get precise, actionable data.

---

## What is CUDA Profiling?

CUDA profiling is the act of **instrumenting and observing** a running CUDA program to collect timing and performance data.

When you run a CUDA program, many things happen simultaneously or in sequence:

1. Memory is allocated on the GPU
2. Data is copied from CPU (host) to GPU (device)
3. Kernels are launched and executed on GPU cores
4. Results are copied back from GPU to CPU
5. GPU memory is freed

Each of these steps takes time, and any of them can be a bottleneck.

A profiler intercepts these operations and records:

- **Start time** and **end time** of each operation
- **Duration** of kernel executions
- **Memory bandwidth** usage
- **GPU hardware counters** (warp efficiency, occupancy, cache hit rates, etc.)

This data is then presented visually or as a report, so you can see exactly where time is being spent.

---

## Why Profiling Matters

Consider this scenario:

> Your matrix multiplication program takes **10 seconds** to run.  
> You spend a week rewriting the kernel to be twice as fast.  
> Your program now takes **9.5 seconds**.

Why? Because the profiler would have shown you that the kernel only took **0.5 seconds** — the other **9.5 seconds** were spent copying data between CPU and GPU. Optimizing the kernel gave almost no benefit.

This is why **profiling must come before optimizing**. It prevents wasted effort and points you directly at the real problem.

### Common scenarios profiling reveals:

| Scenario | What Profiling Shows |
|----------|----------------------|
| Program is slow overall | Memory transfers dominate total runtime |
| Kernel runs but takes too long | Low GPU occupancy or poor memory coalescing |
| GPU utilization looks low | Too many small kernel launches with high overhead |
| Results are correct but slow | Memory access pattern is non-coalesced |
| Scaling poorly with data size | Global memory bandwidth is saturated |

---

## NVTX – Annotating Your Code

### What is NVTX?

**NVTX** (NVIDIA Tools Extension) is a header-only C/C++ API that allows you to insert **named markers and ranges** directly into your source code. These annotations appear visually in the profiler timeline, making it easy to correlate profiling data with specific sections of your program.

Think of NVTX like adding **labels to a map** — the profiler still records everything, but NVTX tells you *what each part means*.

### NVTX Version Notes

| CUDA Version | NVTX Version | How to Link |
|-------------|--------------|-------------|
| CUDA < 10   | NVTX v2      | Requires `-lnvToolsExt` flag |
| CUDA >= 10  | NVTX v3      | Header-only, **no linking needed** |

> **Important:** If you are using CUDA 10 or newer (including CUDA 13.x), include `nvtx3/nvToolsExt.h` and compile **without** `-lnvToolsExt`. The library is entirely in the headers.

### Basic NVTX API

```cpp
#include <nvtx3/nvToolsExt.h>   // NVTX v3 (CUDA >= 10, header-only)

// Mark the start of a named section
nvtxRangePush("Section Name");

    // ... your code here ...

// Mark the end of that section
nvtxRangePop();
```

### Example: Annotated Matrix Multiplication

```cpp
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

void matrixMul(float *A, float *B, float *C, int N) {

    nvtxRangePush("Matrix Multiplication");   // outer range

        nvtxRangePush("Memory Allocation");
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);
        nvtxRangePop();

        nvtxRangePush("H2D Memory Copy");     // Host to Device
        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
        nvtxRangePop();

        nvtxRangePush("Kernel Execution");
        matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        nvtxRangePop();

        nvtxRangePush("D2H Memory Copy");     // Device to Host
        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
        nvtxRangePop();

    nvtxRangePop();  // end Matrix Multiplication
}
```

When you open the profiler output, each `nvtxRangePush`/`nvtxRangePop` pair appears as a labeled colored bar on the timeline. You can immediately see how long each phase takes relative to the others.

### What NVTX does NOT do

- It does **not** speed up your code
- It does **not** change how the GPU executes kernels
- It does **not** add any runtime overhead beyond a simple string push/pop

NVTX is purely a **labeling tool** — it makes the profiling data human-readable.

---

## Nsight Tools Overview

NVIDIA provides two complementary profiling tools as part of the CUDA Toolkit:

| Tool | Primary Purpose | Scope |
|------|----------------|-------|
| **Nsight Systems** (`nsys`) | High-level timeline of entire program | CPU + GPU combined |
| **Nsight Compute** (`ncu`) | Low-level GPU kernel analysis | Individual kernels |

These tools are **complementary, not competing**. The typical workflow is:

1. Use **Nsight Systems** first to find *which* kernel or operation is slow
2. Use **Nsight Compute** to find *why* that kernel is slow

---

## Nsight Systems – System-Wide Analysis

### What it does

Nsight Systems captures a **full timeline** of your program, showing CPU and GPU activity side by side. It records:

- Every CUDA kernel launch and its duration
- All `cudaMemcpy` operations (H2D and D2H)
- Memory allocation and deallocation
- NVTX ranges you defined in code
- CPU function call stacks
- CUDA API calls (cudaMalloc, cudaFree, etc.)
- OS-level thread activity

### How to compile (NVTX v3, CUDA >= 10)

```bash
# No -lnvToolsExt needed for NVTX v3
nvcc main.cu -o program
```

### How to run

```bash
# Basic profiling — generates a .nsys-rep file
nsys profile ./program

# With summary statistics printed to terminal
nsys profile --stats=true ./program

# With a custom output name
nsys profile -o my_profile ./program
```

### Output

Running `nsys profile` produces a `.nsys-rep` file. This can be opened in the **Nsight Systems GUI** on Windows or Linux for visual analysis.

The `--stats=true` flag also prints a human-readable summary table directly to the terminal, showing:

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Min (ns)  Max (ns)  Name
 --------  ---------------  ---------  --------  --------  --------  ----
     85.3       9,530,400          1  9530400   9530400   9530400   cudaMemcpy HtoD
     12.1       1,351,200          1  1351200   1351200   1351200   matrixMulKernel
      2.6         290,400          3    96800     88200    115200   cudaMalloc
```

This immediately shows you which operation consumes the most time.

### What to look for

- **Large H2D/D2H memory copies** — consider reducing data transfers or using pinned memory
- **Gaps between kernel launches** — indicates CPU overhead between GPU operations  
- **Short kernel durations vs long memory transfers** — memory bound program
- **NVTX ranges** — your labeled sections appear as colored bars above the GPU timeline

---

## Nsight Compute – Deep Kernel Analysis

### What it does

While Nsight Systems shows you the *big picture*, Nsight Compute goes **deep into a single kernel** and collects hundreds of hardware-level metrics. It tells you:

- What percentage of peak memory bandwidth is being used
- How many warps are active per SM (occupancy)
- Whether memory accesses are coalesced or scattered
- How much time warps spend stalled waiting for memory
- L1 and L2 cache hit rates
- Instruction throughput

### How to run

```bash
# Basic kernel profiling
ncu ./program

# Save report to file (viewable in Nsight Compute GUI)
ncu -o kernel_report ./program

# Profile a specific kernel by name
ncu --kernel-name matrixMulKernel ./program

# Show specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./program
```

### Permission requirement

By default, `ncu` requires elevated permissions to access GPU hardware counters.

```bash
# Option 1: Run with sudo (quick fix)
sudo ncu ./program

# Option 2: Permanently allow unprivileged profiling (recommended)
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
    > /etc/modprobe.d/nvidia-profiling.conf'
# Then reboot
```

### Key metrics reported by ncu

| Metric | What it means |
|--------|--------------|
| **SM Throughput** | How busy the GPU compute units are (higher = better) |
| **Memory Throughput** | How much of peak memory bandwidth is being used |
| **Occupancy** | Fraction of maximum warps that are active per SM |
| **Warp Stall Reasons** | Why warps are not executing (e.g., waiting for memory) |
| **L1/L2 Hit Rate** | Cache effectiveness — low hit rate means slow random memory access |
| **Achieved IPC** | Instructions executed per clock cycle (higher = better) |

### Example ncu output (simplified)

```
Section: GPU Speed Of Light Throughput
  Compute (SM) Throughput    38.5 %
  Memory Throughput          79.2 %

  → Memory throughput is high. This kernel is memory-bound.
     Consider improving data locality or using shared memory.
```

This tells you the kernel is **memory-bound** — the compute units are underutilized because they keep waiting for data from global memory. The optimization path is clear: use **shared memory tiling** to reuse data.

---

## Nsight Systems vs Nsight Compute

| Feature | Nsight Systems (`nsys`) | Nsight Compute (`ncu`) |
|---------|------------------------|------------------------|
| **Scope** | Entire program | Single kernel |
| **CPU activity** | Yes | No |
| **GPU timeline** | Yes (coarse) | No |
| **NVTX ranges** | Yes | No |
| **Kernel duration** | Yes | Yes |
| **Memory throughput** | Basic | Detailed |
| **Occupancy** | No | Yes |
| **Warp efficiency** | No | Yes |
| **Cache hit rates** | No | Yes |
| **Overhead** | Low | High (replays kernel) |
| **Use case** | "Which part is slow?" | "Why is this kernel slow?" |
| **Output format** | `.nsys-rep` | `.ncu-rep` |

> **Rule of thumb:** Always start with `nsys`. Once you identify the bottleneck kernel, use `ncu` to diagnose it.

---

## Step-by-Step Profiling Workflow

### Step 1 — Write your CUDA program with NVTX annotations

```cpp
#include <nvtx3/nvToolsExt.h>

nvtxRangePush("Phase Name");
// ... code ...
nvtxRangePop();
```

### Step 2 — Compile (no `-lnvToolsExt` for CUDA >= 10)

```bash
nvcc main.cu -o program
```

### Step 3 — Run Nsight Systems for the big picture

```bash
nsys profile --stats=true ./program
```

Read the terminal output: identify which operation dominates runtime.

### Step 4 — Open the visual report (optional)

Transfer the `.nsys-rep` file to a machine with the Nsight Systems GUI installed and open it for a visual timeline.

### Step 5 — Run Nsight Compute on the bottleneck kernel

```bash
sudo ncu --kernel-name mySlowKernel ./program
```

Read the metrics: is it memory-bound? compute-bound? low occupancy?

### Step 6 — Optimize based on evidence

| Profiler finding | Optimization strategy |
|-----------------|----------------------|
| H2D/D2H transfer dominates | Reduce data transfers, use pinned memory, overlap with compute |
| Kernel is memory-bound | Use shared memory tiling, improve access patterns |
| Low occupancy | Reduce register usage, adjust block size |
| Poor cache hit rate | Improve spatial/temporal locality of memory accesses |
| Many small kernels | Fuse kernels to reduce launch overhead |

### Step 7 — Profile again and compare

Optimization is iterative. After each change, profile again to confirm improvement and check for new bottlenecks.

---

## Understanding Profiling Output

### Reading the nsys stats table

```
 Time (%)  Total Time (ns)  Instances  Name
 --------  ---------------  ---------  ----
     85.3       9,530,400          2   [CUDA memcpy Host-to-Device]
     12.1       1,351,200          1   matrixMulKernel
      2.6         290,400          3   [CUDA memcpy Device-to-Host]
```

- **Time (%)** — Percentage of total GPU time this operation consumed
- **Total Time** — Absolute time in nanoseconds
- **Instances** — How many times this operation was called

In this example, **85% of GPU time is memory transfers**. Optimizing the kernel would have minimal impact. The real optimization is reducing or overlapping memory transfers.

### Reading the ncu throughput section

```
Compute (SM) Throughput:   38.5%
Memory Throughput:         79.2%
```

- If **Memory > Compute**: kernel is **memory-bound** → optimize memory access patterns
- If **Compute > Memory**: kernel is **compute-bound** → reduce arithmetic work or increase parallelism
- If both are low: kernel has **latency issues** → increase occupancy or hide latency with more work

---

## Common Performance Bottlenecks

### 1. Excessive Host-Device Memory Transfers

**Problem:** Copying large amounts of data between CPU and GPU is slow (PCIe bandwidth ~16 GB/s vs GPU memory bandwidth ~900 GB/s).

**Fix:** Minimize transfers. Keep data on GPU as long as possible. Use `cudaMemcpyAsync` to overlap transfers with computation.

---

### 2. Uncoalesced Memory Access

**Problem:** When threads in a warp access memory in a non-sequential pattern, the GPU must issue multiple memory transactions instead of one. This severely reduces throughput.

**Bad (stride access):**
```cpp
// Each thread jumps by N — non-coalesced
float val = A[threadIdx.x * N];
```

**Good (sequential access):**
```cpp
// Threads access consecutive addresses — coalesced
float val = A[threadIdx.x];
```

---

### 3. Low Occupancy

**Problem:** If your kernel uses too many registers or too much shared memory per thread block, fewer blocks can be scheduled on each SM. This means fewer warps to hide latency.

**Fix:** Reduce register usage (compiler flag `--maxrregcount`), reduce shared memory per block, or adjust block size.

---

### 4. Global Memory Dependency (No Shared Memory)

**Problem:** Every thread reads from slow global memory on every iteration, even when the same data is reused across threads.

**Fix:** Load shared data into **shared memory** once per block, then have all threads read from it.

```cpp
__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
tile[threadIdx.y][threadIdx.x] = A[row * N + col];
__syncthreads();
// Now use tile[][] instead of A[][]
```

---

### 5. Too Many Small Kernel Launches

**Problem:** Each kernel launch has overhead (~5–10 microseconds). If you launch thousands of tiny kernels, overhead dominates.

**Fix:** Fuse multiple operations into a single kernel where possible.

---

## Key Takeaways

- **Profile before you optimize** — never guess where the bottleneck is
- **NVTX v3 is header-only** — no `-lnvToolsExt` flag needed for CUDA >= 10
- **Nsight Systems** gives the big picture: use it first to find the slow operation
- **Nsight Compute** gives hardware-level details: use it to understand *why* a kernel is slow
- **Most CUDA programs are memory-bound** — optimizing memory access patterns is usually the highest-value improvement
- **Profiling is iterative** — optimize, profile again, repeat

---

## Conclusion

CUDA profiling transforms GPU development from guesswork into an **evidence-driven engineering process**. Rather than randomly trying optimizations and hoping for improvement, profiling gives you precise measurements that tell you exactly what to fix and why.

The toolkit is straightforward:

| Tool | Command | Purpose |
|------|---------|---------|
| NVTX annotations | `nvtxRangePush` / `nvtxRangePop` | Label code sections |
| Nsight Systems | `nsys profile --stats=true ./program` | Find the bottleneck |
| Nsight Compute | `sudo ncu ./program` | Diagnose the bottleneck |

Master these tools and you gain the ability to take a working CUDA program and make it **work at the speed the hardware actually allows**.

---

*CUDA Profiling Guide — CUDA 13.x / Nsight Systems / Nsight Compute*