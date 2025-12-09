# 🚀 CUDA 3‑Month Learning Roadmap

A clean and structured **README.md‑style** version of your CUDA learning plan.

---

## ✅ **Month 1 — Foundations (Weeks 1–4)**

**Goal:** Become comfortable with C/C++ basics and understand how CUDA works.

### 🕒 **Daily Time Split**

* **25–30 min** learning
* **40–60 min** hands‑on coding

---

## **Week 1 — C/C++ Refresh + Setup**

### Learn:

* Variables, loops, functions
* Arrays, pointers
* Basic memory (`malloc`, `free`, stack vs heap)

### Do:

Install:

* NVIDIA Driver (latest)
* CUDA Toolkit
* VS Code + C/C++ + CUDA extensions

Write:

* Array sum in C
* Pointer swap program
* Dynamic array allocation

**✅ Output:** 3 small C programs on GitHub

---

## **Week 2 — Parallel Computing Basics**

### Learn:

* Parallelism vs concurrency
* CPU vs GPU
* SIMD vs MIMD
* Why GPU is fast (many small cores)

### Do:

Write CPU programs:

* Vector addition
* Matrix addition
* Optional: Time them using threads (OpenMP)

**✅ Output:** Clear understanding of why CUDA exists

---

## **Week 3 — First CUDA Programs**

### Learn:

* `__global__`, `__host__`, `__device__`
* `<<<grid, block>>>` syntax
* Thread indexing:
  ```cpp
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  ```

### Do:

* Hello Kernel
* Vector addition on GPU
* Compare CPU vs GPU runtime

**✅ Output:** Your first CUDA GitHub repo 🎉

---

## **Week 4 — CUDA Memory Fundamentals**

### Learn:

* `cudaMalloc`
* `cudaMemcpy`
* `cudaFree`
* Unified memory vs device memory

### Do:

* Matrix addition on GPU
* Try different block sizes
* Intentionally break/fix memory copy errors

**✅ You now officially know CUDA basics**

---

# ✅ **Month 2 — Performance + Real Use Cases (Weeks 5–8)**

**Goal:** Learn how to write *fast* CUDA and optimize performance.

## **Week 5 — Thread Hierarchy + Occupancy**

### Learn:

* Grid vs block vs warp
* Warp size = 32
* What occupancy means

### Do:

Experiment with block sizes:

* 32 threads
* 128 threads
* 256 threads
* 512 threads

**✅ See actual speed differences**

---

## **Week 6 — Shared Memory & Coalescing**

### Learn:

* Global vs shared vs registers
* Memory coalescing
* Bank conflicts

### Do:

Optimize matrix addition:

* Global memory version
* Shared memory version

**✅ This is real CUDA engineering**

---

## **Week 7 — Image Processing Project**

### Learn:

* 2D grids
* Image layout in memory

### Project:

GPU image operations:

* Grayscale conversion
* Blur filter
* Edge detection

Use:

* OpenCV + CUDA
* **or** raw PPM images

**✅ Great portfolio project**

---

## **Week 8 — CUDA + Python (Industry Skill)**

### Learn:

* CuPy
* PyCUDA
* How ML frameworks use CUDA

### Do:

Accelerate Python workloads:

* Large vector ops
* Compare **NumPy vs CuPy**
* Try simple neural network training on GPU

**✅ Now your CUDA connects to AI jobs**

---

# ✅ **Month 3 — Advanced CUDA + Job‑Ready Portfolio (Weeks 9–12)**

## **Week 9 — CUDA Streams & Concurrency**

### Learn:

* CUDA streams
* Async memory copy
* Overlapping compute + memory

### Do:

* Multi‑stream vector addition
* Measure pipeline speedup

---

## **Week 10 — Profiling & Debugging**

### Learn:

* Nsight Compute
* Nsight Systems
* Memory leak detection
* Illegal memory access debugging

### Do:

Profile & optimize:

* Matrix multiplication
* Image processing kernel

**✅ Real GPU engineering workflow**

---

## **Week 11 — Final CUDA Project (Major Portfolio Piece)**

Choose one:

### **Option A — AI Acceleration**

* Speed up neural network inference
* Image classifier optimization
* Object detection preprocessing

### **Option B — Scientific / Engineering**

* Heat diffusion simulation
* Particle simulation
* Fluid grid solver

### **Option C — Embedded + CUDA**

* Sensor data batch processing
* Signal filtering on GPU

**✅ This will be your main CV project**

---

## **Week 12 — Job Prep + Certification + CV**

### Do:

* Clean GitHub repos
* Add READMEs
* Add performance graphs
* CPU vs GPU comparison charts
* Take free NVIDIA DLI CUDA course

Prepare:

* CUDA‑focused CV
* Project explanations

---

# 🎯 **After 3 Months, You Will Be Able To:**

* Write CUDA kernels
* Manage GPU memory
* Optimize performance
* Use CUDA with Python
* Profile GPU workloads
* Build real projects
* Apply for jobs:
  * Junior GPU Engineer
  * ML Engineer (GPU)
  * HPC Trainee
  * AI Infrastructure Intern
