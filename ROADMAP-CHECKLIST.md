# 🚀 CUDA Learning Roadmap Checklist

**Complete guide from CUDA beginner to production-ready GPU engineer**

**Estimated Duration:** 6-12 months depending on pace
**Commitment:** 20-30 hours/week
**Goal:** Production-quality GPU code + portfolio projects

---

## 📊 Learning Path Overview

```
PHASE 1: FOUNDATIONAL KNOWLEDGE (Weeks 1-4)
├── C++ Fundamentals & GPU Architecture
├── CUDA Basics & First Kernels
├── Memory Management
└── Basic Profiling

PHASE 2: CORE COMPETENCIES (Weeks 5-12)
├── Performance Optimization
├── Memory Coalescing & Shared Memory
├── Advanced Patterns (Reduction, Scan)
└── Profiling & Debugging Deep Dive

PHASE 3: PORTFOLIO PROJECTS (Weeks 13-24)
├── Project 1: Matrix Multiplication (GPU Fundamentals)
├── Project 2: Real Application (Images, Physics, ML)
└── Project 3: Performance Analysis Tool

PHASE 4: JOB PREPARATION (Weeks 25-26)
├── GitHub Polish
├── Resume & Certifications
├── Interview Prep
└── Network for Opportunities
```

---

# 📋 PHASE 1: FOUNDATIONS (Weeks 1-4)

## 1: C++ Fundamentals & GPU Basics

### Learn ✍️

- [X] Modern C++ essentials
  - [X] References vs pointers
  - [X] Smart pointers (`unique_ptr`, `shared_ptr`)
  - [X] RAII principle
  - [X] Lambda expressions
  - [X] Auto type deduction
- [X] GPU Architecture (conceptual)
  - [X] GPU = thousands of simple cores vs few powerful CPU cores
  - [X] Latency vs throughput tradeoff
  - [X] GPU memory bandwidth (~200 GB/s) vs CPU (~50 GB/s)
  - [X] Why parallel processing on GPU matters

### Code 💻

- [X] Write 3 C++ programs (no GPU yet)
  - [X] Vector class with operator overloads
  - [X] Smart pointer usage example
  - [X] Template function implementations
- [X] Study existing pointer code

### Checkpoint ✅

- [X] Understand pointer vs reference differences
- [X] Know why GPU is fundamentally different from CPU
- [X] Write memory-safe C++ code with smart pointers

---

## 2: CUDA Setup & First Kernel

### Setup ⚙️

- [X] Install NVIDIA Driver
  - [X] Check current: `nvidia-smi`
  - [X] Download: nvidia.com/download/driverDetails.aspx
  - [X] Verify success: Should show GPU info
- [X] Install CUDA 13.1 Toolkit
  - [X] Download: developer.nvidia.com/cuda-downloads
  - [X] Verify: `nvcc --version` shows CUDA version
- [X] Install Development Tools
  - [X] VS Code
  - [X] C/C++ Extension (ms-vscode.cpptools)
  - [X] CUDA Extension (ms-vscode.cpptools-extension-pack)
  - [X] Makefile support

### Learn ✍️

- [X] CUDA Execution Model (SIMT)
  - [X] Blocks and grids (2D/3D layouts)
  - [X] Threads and warps (32 threads = 1 warp = scheduling unit)
  - [X] `blockIdx`, `threadIdx`, `gridDim`, `blockDim` built-ins
  - [X] Thread indexing: `idx = threadIdx.x + blockIdx.x * blockDim.x`
- [X] CUDA Keywords
  - [X] `__global__` - kernel (GPU function called by CPU)
  - [X] `__device__` - device function (GPU only)
  - [X] `__host__` - host function (CPU only, default)
- [X] Kernel Launch
  - [X] Syntax: `kernel<<<blocks, threads>>>()`
  - [X] Synchronization: `cudaDeviceSynchronize()`
- [X] Error Handling
  - [X] `cudaGetLastError()`
  - [X] Error codes and meanings

### Code 💻

- [X] **Hello CUDA**
  - [X] `__global__` kernel that prints
  - [X] Launch with `<<<1,1>>>()`
  - [X] Add synchronization
  - [X] Compile: `nvcc -arch=sm_86 -o output input.cu`
  - [X] Run and verify output
- [X] **Vector Addition (GPU)**
  - [X] Allocate GPU memory: `cudaMalloc`
  - [X] Copy data: `cudaMemcpy` (HostToDevice)
  - [X] Launch kernel with calculated grid/blocks
  - [X] Copy results back: `cudaMemcpy` (DeviceToHost)
  - [X] Free memory: `cudaFree`
  - [X] Verify against CPU calculation
- [X] **Block Configuration Experiments**
  - [X] Test: <<<1, 256>>>
  - [X] Test: <<<4, 64>>>
  - [X] Test: <<<256, 256>>>
  - [X] Measure performance for each (use `chrono`)

### Checkpoint ✅

- [X] Successfully compile and run first kernel
- [X] Understand grid/block/thread hierarchy deeply
- [X] Know how to allocate, copy, and free GPU memory
- [X] Can calculate thread index from `blockIdx` and `threadIdx`

---

## 3: Memory Fundamentals

### Learn ✍️

- [ ] GPU Memory Types (Critical!)
  - [ ] **Global Memory:** Large (~4-24GB), slow (~100-200 GB/s)
  - [ ] **Shared Memory:** Small (~48-96KB per block), fast
  - [ ] **Registers:** Fastest, per-thread, limited (~256KB per block)
  - [ ] **Constant Memory:** Small, fast for broadcast reads
  - [ ] **Texture Memory:** Cached, good for non-sequential access
- [ ] Memory Coalescing (10x performance impact!)
  - [ ] Sequential access patterns (good): Thread 0→Addr0, Thread 1→Addr1, ...
  - [ ] Strided access (bad): Thread 0→Addr0, Thread 1→Addr16, ...
  - [ ] Random access (very bad): Each thread reads different memory location
  - [ ] GPU memory transactions: 32/64/128 byte units
- [ ] Memory Bandwidth
  - [ ] Calculate achieved bandwidth: (bytes accessed) / (kernel time)
  - [ ] Compare to GPU max bandwidth (from specs)
  - [ ] Target: 80%+ of peak for optimized kernels
- [ ] Host vs Device Naming Convention
  - [ ] Prefix: `h_` for host (CPU) memory
  - [ ] Prefix: `d_` for device (GPU) memory
  - [ ] Makes code self-documenting

### Code 💻

- [ ] **Matrix Addition (Global Memory Only)**
  - [ ] Allocate 2D arrays on CPU
  - [ ] Allocate 2D arrays on GPU
  - [ ] Copy matrices to GPU
  - [ ] Launch 2D kernel (threadIdx.x, threadIdx.y)
  - [ ] Copy results back
  - [ ] Verify results match CPU
  - [ ] Measure execution time
- [ ] **Performance Measurement**
  - [ ] Use `std::chrono` for timing
  - [ ] Measure memory bandwidth achieved
  - [ ] Calculate GFLOPS if compute-bound
  - [ ] Document baseline performance
- [ ] **Memory Access Pattern Experiment**
  - [ ] Sequential access kernel
  - [ ] Strided access kernel (stride = 2, 4, 16)
  - [ ] Random access kernel
  - [ ] Measure performance for each
  - [ ] Document performance differences (should see 10x variation)

### Checkpoint ✅

- [ ] Understand memory hierarchy and latency
- [ ] Know why coalescing matters (10x difference!)
- [ ] Can measure kernel performance accurately
- [ ] Understand global memory bandwidth concept

---

## 4: Error Handling & Profiling Intro

### Learn ✍️

- [ ] CUDA Error Checking
  - [ ] Common error codes: `CUDA_ERROR_INVALID_VALUE`, `CUDA_ERROR_OUT_OF_MEMORY`, etc.
  - [ ] What causes each error type
  - [ ] Best practices: Check ALL CUDA calls
  - [ ] Synchronous vs asynchronous errors
- [ ] Basic Profiling & Timing
  - [ ] `nvidia-smi` for GPU monitoring
  - [ ] GPU memory usage tracking
  - [ ] Power consumption checks
  - [ ] CPU timing with `std::chrono`
  - [ ] Measuring kernel execution time accurately
- [ ] Advanced Compilation
  - [ ] Architecture flags: `-arch=sm_86` (for RTX 30-series), `-arch=sm_80` (A100)
  - [ ] Optimization flags: `-O3`, `-O2`, `-O0` (debug)
  - [ ] Debug symbols: `-g` for debugging
  - [ ] Code generation: `-code=sm_86,compute_86`

### Code 💻

- [ ] **Error Checking Macro**
  ```cpp
  #define CUDA_CHECK(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
      exit(1); \
    } \
  }
  ```
- [ ] **Add Error Checking Everywhere**
  - [ ] Wrap all `cudaMalloc`, `cudaMemcpy`, `cudaFree`
  - [ ] Wrap all kernel launches
  - [ ] Wrap `cudaDeviceSynchronize`
- [ ] **Complete Vector Addition with Production Code**
  - [ ] Proper error handling on ALL CUDA calls
  - [ ] Performance timing with high-resolution clock
  - [ ] GPU memory monitoring with `nvidia-smi` in script
  - [ ] Input validation (check sizes > 0)
- [ ] **Debugging Practice**
  - [ ] Intentionally pass invalid GPU pointer
  - [ ] Observe error message
  - [ ] Intentionally allocate too much memory
  - [ ] Catch and handle out-of-memory error
  - [ ] Intentionally cause warp divergence (to understand it)

### Checkpoint ✅

- [ ] All programs have comprehensive error handling
- [ ] Can measure performance with timing
- [ ] Understand CUDA error codes and what causes them
- [ ] Compiler architecture flags appropriate for your GPU

### 🎓 End of Phase 1

✅ Write correct, memory-safe CUDA code
✅ Understand thread hierarchy and indexing
✅ Know all memory operations
✅ Can measure basic performance
❌ NOT optimized - just functional

---

# 🔥 PHASE 2: PERFORMANCE OPTIMIZATION (Weeks 5-12)

## 5: Occupancy & Block Sizing

### Learn ✍️

- [ ] GPU Occupancy (The Utilization Metric)
  - [ ] Definition: % of GPU cores actually doing useful work
  - [ ] Factors affecting occupancy:
    - Block size (# threads)
    - Register usage per thread
    - Shared memory usage per block
  - [ ] Why occupancy matters (more threads = better latency hiding)
  - [ ] When high occupancy doesn't mean high performance (memory-bound kernels)
- [ ] Block Size Selection
  - [ ] Multiples of warp size (32): 32, 64, 128, 256, 512
  - [ ] Common choices: 128, 256 threads per block
  - [ ] Tradeoffs:
    - Larger blocks: Higher occupancy, more synchronization
    - Smaller blocks: Lower occupancy, easier to synchronize
  - [ ] Your GPU's limits (check specs for max threads per block)

### Code 💻

- [ ] **Vector Addition with Block Size Experiments**
  - [ ] Test block sizes: 32, 64, 128, 256, 512 threads
  - [ ] For each: Measure execution time
  - [ ] Calculate occupancy for each configuration
  - [ ] Create performance table: Block Size → Time → Occupancy
  - [ ] Document which is fastest
  - [ ] Plot performance vs block size
- [ ] **Reduction Kernel (Important Pattern!)**
  - [ ] Sum all array elements on GPU
  - [ ] Basic version (no optimization)
  - [ ] Try different block sizes
  - [ ] Measure throughput (elements/second)
  - [ ] Compare to theoretical peak

### Checkpoint ✅

- [ ] Know optimal block size for your GPU
- [ ] Understand occupancy tradeoffs
- [ ] Can calculate occupancy for any configuration
- [ ] Know when occupancy matters and when it doesn't

---

## 6: Memory Coalescing & Shared Memory

### Learn ✍️

- [ ] Memory Coalescing Deep Dive
  - [ ] GPU memory transactions: 32, 64, or 128 byte units
  - [ ] Warp (32 threads) accesses memory simultaneously
  - [ ] Good pattern: All 32 threads read consecutive bytes
  - [ ] Bad pattern: Strided or random access
  - [ ] Bank conflicts: Multiple threads accessing same memory bank
- [ ] Shared Memory Optimization
  - [ ] Declaration: `__shared__ int data[256];`
  - [ ] Per-block allocation (~48-96KB available)
  - [ ] Synchronization: `__syncthreads()` (critical!)
  - [ ] Shared memory is 10-100x faster than global memory
- [ ] Bank Conflicts in Shared Memory
  - [ ] Shared memory organized into banks (32 banks typical)
  - [ ] Bank conflict = multiple threads access same bank = serialization
  - [ ] Solution: Padding or careful access patterns

### Code 💻

- [ ] **Matrix Transpose (Classic Coalescing Example)**
  - [ ] Naive version: Read column-wise, write row-wise (bad coalescing)
  - [ ] Measure performance
  - [ ] Shared memory optimized version:
    - Read into shared memory (coalesced)
    - Write out with synchronization (coalesced)
  - [ ] Measure performance (should be 3-5x faster!)
  - [ ] This is a classic optimization example
- [ ] **Matrix Multiplication**
  - [ ] Global memory version
  - [ ] Measure throughput (GFLOPS)
  - [ ] Shared memory tile blocking version
  - [ ] Compare performance
  - [ ] Profile memory bandwidth with Nsight Compute
- [ ] **Memory Pattern Analysis**
  - [ ] Write kernels with different access patterns:
    - Sequential
    - Strided (stride = 2, 4, 16)
    - Random
  - [ ] Measure bandwidth for each
  - [ ] Identify coalescing issues in your code
  - [ ] Fix and re-measure improvements

### Checkpoint ✅

- [ ] Identify coalescing issues in code (without profiler)
- [ ] Write shared memory optimized kernels
- [ ] Measure 5-10x improvement from optimization
- [ ] Understand bank conflicts and how to avoid them

---

## 7-8: Reduction & Scan Patterns (NVIDIA Favorites!)

### Learn ✍️

- [ ] **Reduction Pattern (80% of CUDA workloads!)**
  - [ ] Problem: Find max/min/sum of all elements
  - [ ] Naive approach: Use atomic operations (slow)
  - [ ] Work-efficient reduction:
    - Level 1: Each thread sums 2 elements
    - Level 2: Each thread sums 4 elements
    - Continue until 1 thread has result
  - [ ] Bank-conflict free reduction: Padding in shared memory
  - [ ] Warp reduction: Use `__shfl_down_sync()` for final steps
  - [ ] Common mistakes: Wrong synchronization, race conditions
- [ ] **Scan (Prefix Sum) Pattern**
  - [ ] Problem: Compute prefix sum/product of array
  - [ ] Inclusive scan: output[i] = sum(input[0..i])
  - [ ] Exclusive scan: output[i] = sum(input[0..i-1])
  - [ ] Hillis-Steele algorithm: Multiple passes, ~O(log n) steps
  - [ ] Work-efficient (Blelloch) algorithm: Fewer operations
  - [ ] Applications: Stream compaction, parallel quicksort, counting sort
- [ ] **Atomic Operations**
  - [ ] `atomicAdd`, `atomicMax`, `atomicCAS`, etc.
  - [ ] When to use (rarely, they're slow)
  - [ ] Synchronization guarantees they provide

### Code 💻

- [ ] **Efficient Reduction Implementation**
  - [ ] Step 1: Baseline (no optimization)
    - Basic reduction using global memory and synchronization
  - [ ] Step 2: Remove bank conflicts
    - Pad shared memory to avoid banks
    - Re-measure performance improvement
  - [ ] Step 3: Optimize for your GPU
    - Tune tile size, block size, etc.
    - Re-measure each time
  - [ ] Create table: Optimization Effect → Performance Improvement
  - [ ] Should show cumulative 5-10x improvement
- [ ] **Inclusive Scan**
  - [ ] Implement Hillis-Steele algorithm
  - [ ] Test on various input sizes (1K, 10K, 100K, 1M)
  - [ ] Verify correctness vs CPU scan
  - [ ] Measure throughput (elements/second)
- [ ] **Exclusive Scan**
  - [ ] Similar to inclusive, but with different indexing
  - [ ] Verify correctness carefully
  - [ ] Test edge cases: single element, power-of-2 vs non-power-of-2
- [ ] **Performance Documentation**
  - [ ] Create performance table showing each optimization step
  - [ ] Plot: Problem Size → Throughput
  - [ ] Compare: CPU vs GPU
  - [ ] This becomes portfolio material!

### Checkpoint ✅

- [ ] Implement efficient reduction kernel
- [ ] Implement both scan variants
- [ ] Understand fundamental CUDA design patterns
- [ ] Measure and document optimization progression

---

## 9-12: Streams, Advanced Profiling & Integration

### Learn ✍️

- [ ] **CUDA Streams**
  - [ ] Default stream (stream 0): Sequential execution
  - [ ] Custom streams: Enable parallel execution
  - [ ] Creation: `cudaStreamCreate(&stream)`
  - [ ] Usage: Pass stream to all CUDA calls
  - [ ] Stream synchronization: `cudaStreamSynchronize(stream);`
  - [ ] Benefits: Overlap compute and memory transfer
- [ ] **Asynchronous Operations**
  - [ ] `cudaMemcpyAsync()` - async memory copy
  - [ ] `cudaMemcpy()` vs `cudaMemcpyAsync()`: Blocking vs non-blocking
  - [ ] Kernel-memory overlap: Kernel on GPU while transfer happens
  - [ ] CUDA Graphs (advanced): Record sequence of operations for replay
- [ ] **NVTX (NVIDIA Tools Extension)**
  - [ ] Purpose: Add named markers to code for profiler
  - [ ] Linking: NVTX v3 (CUDA >= 10) is header-only, no linking needed
  - [ ] API: `nvtxRangePushA("name")`, `nvtxRangePop()`
  - [ ] Use in Nsight visualization of timeline
- [ ] **Nsight Systems (System-wide Profiling)**
  - [ ] Captures: GPU operations, memory transfers, CPU-GPU sync
  - [ ] Timeline view: See when GPU is busy vs idle
  - [ ] Identifies: Kernel launch overhead, PCIe transfer bottlenecks
  - [ ] Usage: `nsys profile ./program`
- [ ] **Nsight Compute (Kernel Deep-Dive)**
  - [ ] Captures: Per-kernel detailed metrics
  - [ ] Metrics: Occupancy, warp efficiency, memory coalescing, bank conflicts
  - [ ] Identifies: Specific optimization opportunities
  - [ ] Usage: `ncu ./program`
- [ ] **Key Performance Metrics**
  - [ ] Occupancy: % of GPU utilized (aim for >70%)
  - [ ] Warp efficiency: % of threads actually computing
  - [ ] Memory coalescing: % of efficient memory accesses
  - [ ] Memory throughput: Achieved vs theoretical peak
  - [ ] Register spill: Overflow to slower memory (bad!)

### Code 💻

- [ ] **Multi-Stream Vector Addition**
  - [ ] Create 4 independent streams
  - [ ] Process data in chunks
  - [ ] Each stream: copy → compute → copy result
  - [ ] Streams run independently in parallel
  - [ ] Measure total time vs sequential
  - [ ] Document speedup (should be 2-3x)
- [ ] **Overlapped Compute + Memory Transfer**
  - [ ] Pattern: Copy chunk 1 while computing on chunk 0
  - [ ] Kernel → async memcpy → kernel pattern
  - [ ] Measure time reduction
  - [ ] Document pipeline efficiency
- [ ] **NVTX Annotation**
  - [ ] Add markers to matrix multiplication project
  - [ ] Mark: Memory allocation, H2D copy, kernel, D2H copy
  - [ ] Visualize in Nsight Systems
- [ ] **Profile with Nsight Systems**
  - [ ] Run on matrix multiplication kernel
  - [ ] Identify time spent on compute vs transfers
  - [ ] Analyze timeline for inefficiencies
- [ ] **Profile with Nsight Compute**
  - [ ] Run on several kernels
  - [ ] Check occupancy (use calculator to verify)
  - [ ] Check memory throughput
  - [ ] Identify optimization opportunities
- [ ] **Optimization Based on Profiling**
  - [ ] Profile baseline kernel
  - [ ] Make one optimization
  - [ ] Re-profile and document improvement
  - [ ] Repeat 2-3 times
  - [ ] Build case study: "Optimization Journey"

### Checkpoint ✅

- [ ] Profile kernels professionally with Nsight tools
- [ ] Understand performance bottlenecks from metrics
- [ ] Measure and document performance improvements
- [ ] Know when optimization targets actually matter

### 🎓 End of Phase 2

✅ Optimize CUDA kernels to 5-10x speedup
✅ Master memory hierarchy and coalescing
✅ Implement reduction and scan (fundamental patterns)
✅ Profile and improve performance systematically
✅ Production-level optimization skills

---

# 💼 PHASE 3: PORTFOLIO PROJECTS (Weeks 13-24)

## Project 1: GPU Matrix Multiplication (6 weeks)

**Why Build This:** NVIDIA uses matrix multiply for everything (AI, physics, graphics)

### Milestone 1: CPU Reference (Week 13) — 15 hours

- [ ] Implement matrix multiplication on CPU
  - [ ] Standard O(N³) algorithm
  - [ ] Handle rectangular matrices
- [ ] Verify correctness (small test case by hand)
- [ ] Measure baseline performance
- [ ] Document: Algorithm, time complexity, achieved GFLOPS

### Milestone 2: Naive GPU (Week 14) — 15 hours

- [ ] Implement basic GPU kernel
  - [ ] One thread = one output element
  - [ ] Global memory only (no optimization)
- [ ] Measure performance (likely slower than CPU initially!)
- [ ] Document: Why it's slower, bandwidth achieved

### Milestone 3: Shared Memory (Week 15) — 15 hours

- [ ] Tile-based matrix multiplication
  - [ ] 2D shared memory buffers
  - [ ] Load tiles cooperatively
  - [ ] Compute with tiles from shared memory
- [ ] Optimize tile size for:
  - [ ] Occupancy
  - [ ] Shared memory usage
  - [ ] Cache behavior
- [ ] Measure speedup (target 3-5x vs naive)

### Milestone 4: Memory Coalescing (Week 16) — 15 hours

- [ ] Ensure all memory accesses are coalesced
  - [ ] Global memory reads (coalesced)
  - [ ] Shared memory accesses (avoid bank conflicts)
  - [ ] Global memory writes (coalesced)
- [ ] Use profiler to verify coalescing %
- [ ] Measure additional speedup

### Milestone 5: Advanced Optimization (Week 17) — 15 hours

- [ ] Reduce register pressure
- [ ] Increase occupancy
- [ ] Optimize for your specific GPU
- [ ] Test different block configurations
- [ ] Measure cumulative speedup

### Milestone 6: Tensor Cores (Week 18) — Optional/Advanced

- [ ] If GPU supports tensor cores (RTX 2000+, A100, etc.)
- [ ] Use `wmma` (warp matrix matrix multiply) operations
- [ ] Measure 20-50x total improvement possibility

### Deliverables for Portfolio 📦

- [ ] **Source Code**
  - [ ] `matmul_cpu.cpp` - CPU reference
  - [ ] `matmul_v1.cu` - Naive GPU version
  - [ ] `matmul_v2.cu` - Shared memory optimized
  - [ ] `matmul_final.cu` - Fully optimized
  - [ ] `helper.h` - Shared utilities and macros
- [ ] **Performance Documentation**
  - `performance.txt` or `performance.csv`:
    ```
    Version | Matrix Size | Time (ms) | GFLOPS | Speedup vs CPU
    CPU     | 1024x1024   | 150       | 14.4   | 1.0x
    GPU v1  | 1024x1024   | 200       | 10.7   | 0.75x (slower!)
    GPU v2  | 1024x1024   | 50        | 42.9   | 3.5x
    GPU v3  | 1024x1024   | 35        | 61.0   | 4.3x
    GPU Final| 1024x1024  | 25        | 85.5   | 6.0x
    ```
  - Performance graph: Version vs GFLOPS
- [ ] **README.md**
  - [ ] Algorithm explanation
  - [ ] Optimization steps and their impact
  - [ ] Performance graphs
  - [ ] Compilation instructions
  - [ ] How to run: `./matmul 1024` (creates 1024x1024 matrices)
  - [ ] Verified correctness statement
  - [ ] Profiling analysis for final version
  - [ ] Achieved memory bandwidth %
  - [ ] Achieved occupancy %

### GitHub Quality Checklist

- [ ] Clean, readable code with comments
- [ ] No memory leaks (verify with valgrind if available)
- [ ] Proper error handling on all CUDA calls
- [ ] .gitignore excludes binaries and build artifacts
- [ ] Good commit history showing optimization progression

**💡 This project alone can get you interviews!**

---

## Project 2: Real-World Application (8 weeks)

Choose ONE:

### Option A: Image Processing Pipeline

**What you'll build:** Blur + Sharpen + Edge Detection on GPU

**Weeks 19-20: Core Algorithms (30 hours)**

- [ ] Blur kernel (2D convolution with Gaussian kernel)
  - [ ] Handle boundary conditions
  - [ ] Optimize with shared memory
  - [ ] Test on small images
- [ ] Sharpen kernel
  - [ ] Unsharp mask or similar algorithm
  - [ ] Optimization
- [ ] Sobel edge detection
  - [ ] X and Y gradients
  - [ ] Combine for edge magnitude

**Weeks 21-22: GPU Optimization (30 hours)**

- [ ] Each kernel with shared memory optimization
- [ ] Memory coalescing verification
- [ ] Profile with Nsight Compute

**Weeks 23-24: Full Pipeline (30 hours)**

- [ ] Process real images (use OpenCV or PPM format)
- [ ] Chain filters: Input → Blur → Sharpen → Edges → Output
- [ ] Performance benchmarks: Time vs image size
- [ ] GPU vs CPU comparison
- [ ] Visualization: Show input and output images

---

### Option B: Physics Simulation (Heat Diffusion)

**What you'll build:** 2D heat equation solver on GPU

**Weeks 19-20: Core Solver (30 hours)**

- [ ] Implement Jacobi method
  - [ ] Boundary conditions
  - [ ] Convergence criteria
  - [ ] CPU reference
- [ ] Verify against analytical solution
- [ ] Test with different initial conditions

**Weeks 21-22: GPU Optimization (30 hours)**

- [ ] GPU kernel with shared memory
- [ ] Ping-pong buffers for stencil operation
- [ ] Memory layout for coalescing
- [ ] Profile and optimize

**Weeks 23-24: Advanced & Visualization (30 hours)**

- [ ] Faster solvers (multi-grid method with CUDA)
- [ ] Visualization: Heatmap of temperature field
- [ ] Performance: Solve time vs grid size (100x100 to 10000x10000)
- [ ] Speedup: GPU vs CPU for large grids

---

### Option C: Neural Network Inference

**What you'll build:** GPU-accelerated matrix operations for NN layers

**Weeks 19-20: Core Operations (30 hours)**

- [ ] Matrix multiply (from Project 1!)
- [ ] ReLU activation: `output[i] = max(0, input[i])`
- [ ] Softmax activation (for output layer)
- [ ] BatchNorm (if you want complexity)

**Weeks 21-22: Batch Processing (30 hours)**

- [ ] Inference on batches of inputs
- [ ] Different batch sizes: 1, 32, 256, 1024
- [ ] Performance curves: Batch size vs throughput

**Weeks 23-24: Full Network (30 hours)**

- [ ] Load pre-trained model weights
- [ ] Multi-layer forward pass on GPU
- [ ] Benchmark against TensorFlow/PyTorch
- [ ] Performance: Batch size vs latency

---

## Project 3: Performance Analysis Tool (4-6 weeks)

**What you'll build:** Automated benchmark and analysis suite

**Features:**

- [ ] Test suite of kernels (reduction, scan, transpose, etc.)
- [ ] Auto-tune block sizes
- [ ] Report performance metrics
- [ ] Compare optimization versions
- [ ] Generate performance reports (text or HTML)

**Provides:**

- [ ] Sample kernels for reference
- [ ] Template for profiling
- [ ] Analysis automation

---

## 🎯 End of Phase 3

✅ 2-3 complete projects with clear progression
✅ Each shows optimization evolution
✅ Profiling data and performance graphs
✅ Production-quality, documented code
✅ Portfolio ready for job applications

---

# 🎯 PHASE 4: JOB PREPARATION (Weeks 25-26)

## GitHub Polish

- [ ] **Code Quality Review**
  - [ ] All files compile without warnings
  - [ ] No memory leaks (use valgrind or sanitizers)
  - [ ] Proper error handling everywhere
  - [ ] Clean, readable, well-commented code
  - [ ] Consistent naming conventions
- [ ] **Repository Structure**
  - [ ] Top-level README describing all projects
  - [ ] Individual README per project
  - [ ] /src, /include, /Makefile organization
  - [ ] .gitignore excludes: *.o, binaries, build/, data/
- [ ] **Documentation**
  - [ ] Algorithm explanations
  - [ ] Performance graphs and tables
  - [ ] Compilation instructions
  - [ ] How to run each program
  - [ ] Verification/testing notes
- [ ] **Commit History**
  - [ ] Clear commit messages showing optimization journey
  - [ ] Not one big mega-commit
  - [ ] Shows progression and learning

## Resume Updates

**CUDA Skills Section:**

- [ ] GPU Computing with CUDA
- [ ] GPU Memory Optimization
- [ ] Kernel Optimization & Profiling
- [ ] NVIDIA Nsight (Systems & Compute)
- [ ] Parallel Algorithm Design

**Projects Section - Be Specific:**

```
GPU Matrix Multiplication Accelerator (6 weeks)
• Implemented optimized CUDA kernels achieving 6x speedup through 
  shared memory tiling and memory coalescing
• Profiled with Nsight Compute to maintain 85% peak memory bandwidth
• Supports matrices up to 8192x8192, 120+ GFLOPS on RTX 3060

2D Image Processing Pipeline
• GPU-accelerated blur, sharpen, edge detection filters
• Achieved 8x speedup vs CPU implementation on 4K images
• Optimized memory transfers to overlap with computation

[More projects...]
```

## Certifications

- [ ] **NVIDIA DLI Fundamentals** (Free!)
  - [ ] URL: https://courses.nvidia.com/courses/course-v1:DLI+C-AC-01+A1/about
  - [ ] Duration: 2-3 days
  - [ ] Certificate: Professional credential
  - [ ] Shows foundational knowledge

## Interview Prep

### Technical Interview Topics

**Know These Cold:**

1. [ ] Explain each project optimization in detail
2. [ ] CUDA thread hierarchy (grids, blocks, threads)
3. [ ] Memory coalescing and its impact
4. [ ] Occupancy and when it matters
5. [ ] How to profile and identify bottlenecks
6. [ ] Shared memory bank conflicts
7. [ ] Reduction and scan patterns
8. [ ] CUDA streams and async operations

### Common Interview Questions

1. **"Walk me through how you optimized matrix multiplication"**

   - Describe: Naive → Shared memory → Coalescing → Occupancy tuning
   - Metrics: Show 6x speedup from profiling data
   - Explain: Why each step helped
2. **"What's the memory bandwidth achieved in your project?"**

   - Calculate: (bytes transferred) / (kernel time in seconds) / 1e9 = GB/s
   - Compare: To peak GPU bandwidth (e.g., 360 GB/s for RTX 3060)
   - Achievement: Aim for 80%+ of peak for optimized kernels
3. **"Explain thread coalescing and why it matters"**

   - Answer: Sequential accesses are grouped into single memory transaction
   - Non-coalesced: Each access is separate transaction (10x slower!)
   - Your project: Showed 3-5x improvement by fixing coalescing
4. **"How would you optimize [given kernel]?"**

   - Approach:
     1. Understand computation (what the kernel does)
     2. Profile baseline (what's the bottleneck?)
     3. Memory-bound? → optimize memory access, use shared memory
     4. Compute-bound? → reduce instructions, better registers
     5. Reload balance? → mix both strategies
     6. Re-measure after each change
5. **"You have limited shared memory. How do you decide what goes there?"**

   - Answer: Put data that's re-used multiple times
   - Example: Matrix mul - load tiles of A and B that's used 256 times
   - Tradeoff: Shared memory reduces occupancy (blocks can't launch as many)
6. **"Tell me about a technical challenge you solved"**

   - Story: "In my matrix multiply project, initial version achieved only 10% peak bandwidth"
   - Challenge: Understood it was memory coalescing issue
   - Solution: Restructured tile loading to access memory sequentially
   - Result: 8x bandwidth improvement, 6x speedup overall

### Behavioral Interview

- [ ] **"Why do you want to work at NVIDIA?"**

  - "GPU computing is the future of computing. NVIDIA is leading this space."
  - "Your hardware enables the AI revolution. I want to optimize software for it."
  - "The engineering challenges at NVIDIA are world-class."
- [ ] **"Tell us about your most complex project"**

  - Use your main project (Matrix Mul or Application)
  - Describe process: baseline → profiling → optimization → validation
  - Metrics: Speedup achieved, engineering rigor
- [ ] **"How do you approach performance problems?"**

  - "Measure first, optimize second. Never optimize by guessing."
  - "Profile to identify bottleneck. Focus optimization efforts there."
  - "Verify improvement with re-profiling."
  - "Document for future reference."

## LinkedIn & Networking

- [ ] Update profile with CUDA/GPU skills
- [ ] Add portfolio projects
- [ ] Connect with NVIDIA engineers
- [ ] Join communities:
  - NVIDIA forums (forums.developer.nvidia.com)
  - GitHub discussions on GPU computing
  - Hacker News GPU threads
  - Reddit: r/MachineLearning, r/CUDA

## Application Strategy

### Target Roles at NVIDIA

- **GPU Software Engineer (Entry-level)**

  - CUDA optimization for drivers
  - Kernel development
  - Performance engineering
- **GPU Research Engineer Intern**

  - Academic background helpful
  - GPU computing research
- **AI Infrastructure Engineer**

  - CUDA + ML frameworks
  - Good if you have NN project

### How to Apply

1. Go to: nvidia.com/en-us/about-nvidia/careers/
2. Search for entry-level or intern GPU positions
3. In application/cover letter:
   - Reference specific CUDA projects
   - Mention performance improvements (speedups)
   - Show profiling expertise
   - Express genuine interest in GPU optimization

---

## 🎓 Success Verification Checklist

### Technical Knowledge ✅

- [ ] Explain GPU architecture from first principles
- [ ] Describe SIMT execution model
- [ ] Calculate thread index from block/grid indices
- [ ] Explain memory coalescing and its impact
- [ ] Know bank conflict resolution strategies
- [ ] Understand occupancy factors and importance

### Portfolio Quality ✅

- [ ] 2-3 projects on GitHub (production code quality)
- [ ] Each project shows optimization progression
- [ ] Performance graphs and metrics included
- [ ] Profiling data documented
- [ ] Code is clean, commented, well-organized
- [ ] README is comprehensive and clear

### Practical Skills ✅

- [ ] Write correct CUDA code without references
- [ ] Optimize kernel from "working" to "fast"
- [ ] Profile code systematically
- [ ] Identify and fix performance bottlenecks
- [ ] Explain every optimization decision
- [ ] Know GPU constraints and realistic targets

### Confidence & Readiness ✅

- [ ] Comfortable discussing GPU computing for 30+ minutes
- [ ] Excited about writing new kernels
- [ ] Specific reasons for wanting NVIDIA job
- [ ] Can handle technical interview questions
- [ ] Projects better than 90% of college portfolios

---

## 🚀 SUCCESS TIMELINE

- **Weeks 1-4:** Learn CUDA fundamentals (30 hours)
- **Weeks 5-12:** Master optimization techniques (60 hours)
- **Weeks 13-24:** Build portfolio projects (120+ hours)
- **Weeks 25-26:** Polish and prepare job application (20 hours)

**Total Time: ~230 hours (5-6 months at 20-30 hrs/week)**

**Expected Outcome: Job-ready for entry-level GPU Software Engineer role at NVIDIA**

---

## ✨ Final Wisdom

### Do This ✅

- Profile EVERY change (measure vs guessing)
- Commit with clear messages showing progression
- Write production-quality code
- Document decisions and tradeoffs
- Test thoroughly (correctness first, speed second)
- Show your work (GitHub tells the story)
- Be proud - these are impressive projects!

### Don't Do This ❌

- Skip profiling to "save time" (wastes time!)
- Copy code without understanding
- Only show final result (show the journey)
- Neglect code quality for speed
- Send generic job applications
- Undersell your projects
- Give up on hard problems

---

**Created:** CUDA Learning Roadmap with Practical Checklists
**Focus:** Actionable steps, portfolio building, NVIDIA job readiness
**Status:** Ready to execute! 🚀

**Last Updated:** March 2026
