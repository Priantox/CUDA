# CUDA Basics: Host vs Device (Simple Explanation)

This README explains the **basic CUDA concepts** in a simple and beginner-friendly way.

---

## 1. Host vs Device

### Host (CPU)
- The **CPU** is called the **Host**
- Uses **RAM sticks** on the motherboard
- Runs normal C/C++ code
- Controls program flow and launches GPU work

👉 Think of the CPU as the **manager**

---

### Device (GPU)
- The **GPU** is called the **Device**
- Uses **VRAM** (memory on the GPU chip)
- Handles **massively parallel computation**
- Best for heavy math and data-parallel tasks

👉 Think of the GPU as **thousands of workers**

---

## 2. CUDA Program Flow (High Level)

A typical CUDA program follows these steps:

1. Copy input data from **Host → Device**
2. Run GPU code (kernel) on the data
3. Copy results from **Device → Host**
4. Use or display the result on CPU

👉 CPU prepares work  
👉 GPU processes data  
👉 CPU collects results

---

## 3. Host vs Device Variable Naming

Common naming convention:

- `h_A` → Variable stored on **Host (CPU)**
- `d_A` → Variable stored on **Device (GPU)**

Example:
- `h_A` = matrix in RAM
- `d_A` = matrix copied to GPU VRAM

This makes code **easier to read and debug**.

---

## 4. CUDA Function Types

### `__global__` — GPU Kernel (Called by CPU)
- Runs on **GPU**
- Called from **CPU**
- Does not return values
- Modifies data passed as arguments

Example use:
- Matrix multiplication
- Vector addition
- Image processing

👉 CPU says: *“GPU, run this code in parallel”*

---

### `__device__` — GPU Helper Function
- Runs **only on GPU**
- Can be called **only by GPU code**
- Used as a helper or utility function

Example:
- Apply scalar mask
- Activation functions
- Small math operations

👉 Like calling a library function inside GPU code

---

### `__host__` — CPU-Only Function
- Runs **only on CPU**
- Same as normal C/C++ functions
- No GPU involvement

👉 Default behavior if no CUDA keyword is used

---

## 5. Quick Comparison Table

| Keyword | Runs On | Called By |
|------|-------|---------|
| `__host__` | CPU | CPU |
| `__global__` | GPU | CPU |
| `__device__` | GPU | GPU |

---

## 6. Simple Mental Model

- **CPU (Host)** = Manager 🧠  
- **GPU (Device)** = Workers ⚙️  
- **`__global__`** = Manager assigns work  
- **`__device__`** = Workers helping workers  
- **`__host__`** = Manager-only tasks  

---

# CUDA Memory Copy and Free Explained

CUDA provides functions to transfer data between the CPU (host) and GPU (device), as well as to manage memory on the GPU. Here are the key concepts explained simply:

## cudaMemcpy

`cudaMemcpy` is used to copy data between host (CPU) and device (GPU) memory. It can copy in three main ways:

### 1. Host to Device (CPU to GPU)
- **Purpose:** Send data from your computer's main memory (RAM) to the GPU's memory.
- **Use case:** Before running a GPU computation, you need to send input data to the GPU.
- **Flag:** `cudaMemcpyHostToDevice`

### 2. Device to Host (GPU to CPU)
- **Purpose:** Bring results or data back from the GPU to your computer's main memory.
- **Use case:** After GPU computation, you want to access the results on the CPU.
- **Flag:** `cudaMemcpyDeviceToHost`

### 3. Device to Device (GPU to GPU)
- **Purpose:** Copy data from one location in GPU memory to another (can be the same GPU or between different GPUs).
- **Use case:** Rearranging or duplicating data on the GPU, or transferring data between GPUs.
- **Flag:** `cudaMemcpyDeviceToDevice`

## Example Usage
```c
cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice); // CPU → GPU
cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost); // GPU → CPU
cudaMemcpy(devicePtr2, devicePtr1, size, cudaMemcpyDeviceToDevice); // GPU → GPU
```

## cudaFree

- **Purpose:** Frees (deletes) memory that was allocated on the GPU.
- **Why:** Just like you use `free()` in C to release RAM, you use `cudaFree()` to release GPU memory and avoid memory leaks.
- **Example:**
```c
cudaFree(devicePtr); // Frees memory on the GPU
```

## Summary Table
| Direction         | Function Flag             | Description                |
|-------------------|--------------------------|----------------------------|
| Host → Device     | cudaMemcpyHostToDevice   | CPU to GPU                 |
| Device → Host     | cudaMemcpyDeviceToHost   | GPU to CPU                 |
| Device → Device   | cudaMemcpyDeviceToDevice | GPU to GPU (or between GPUs)|

- Always free GPU memory with `cudaFree` when done.
- Use the correct flag for the direction of data transfer.

---
**Tip:** Always check for errors after calling CUDA functions to ensure your program runs correctly!

---

# CUDA Compilation: Host and Device Code Explained

## nvcc Compiler
- `nvcc` is the CUDA compiler that handles both CPU (host) and GPU (device) code in your program.

## Host Code
- This is the code that runs on the CPU (your computer's main processor).
- `nvcc` compiles host code into a standard x86 binary (like any C/C++ program).
- Host code is modified to launch and manage GPU kernels.

## Device Code
- This is the code that runs on the GPU (the CUDA kernels and device functions).
- `nvcc` compiles device code into **PTX** (Parallel Thread Execution), which is an intermediate assembly language for NVIDIA GPUs.
- PTX is stable across multiple GPU generations, so the same code can work on different GPUs.

## JIT (Just-In-Time) Compilation
- When your program runs, the PTX code is compiled by the GPU driver into native GPU instructions (machine code) for your specific GPU.
- This process is called JIT compilation.
- JIT allows for **forward compatibility**: PTX code can run on future GPUs without recompiling your source code.

## Summary Table
| Stage                | Output         | Runs On         | Purpose                                  |
|----------------------|---------------|-----------------|------------------------------------------|
| Host Compilation     | x86 binary    | CPU             | Launches and manages GPU work            |
| Device Compilation   | PTX           | GPU (via driver)| GPU kernels, portable across generations |
| JIT Compilation      | GPU machine code | GPU          | Optimized for your specific GPU          |

---
**In short:**
- `nvcc` compiles both CPU and GPU code.
- Host code runs on CPU, device code runs on GPU.
- Device code is first compiled to PTX, then JIT-compiled to GPU instructions at runtime.
- This makes CUDA programs portable and future-proof!

---

## What is x86? (Simple Explanation)

- **x86** is a family of computer processors and the instruction set they use.
- Most desktop and laptop CPUs (like Intel and AMD) are x86 processors.
- When you compile C/C++ code for your computer, it becomes an **x86 binary**—a program that your CPU can understand and run.
- x86 is called an "architecture" because it defines how the CPU works and what instructions it understands.

👉 In short: **x86 means normal PC CPU code** (not GPU code).

---

# CUDA Hierarchy: Grids, Blocks, Threads, and Warps (Simple Explanation)

## CUDA Hierarchy
- A **kernel** is a function that runs on the GPU.
- When you launch a kernel, it runs as many **threads** in parallel.
- **Threads** are grouped into **Blocks** (also called Thread Blocks).
- **Blocks** are grouped into a **Grid**.
- So, a kernel is executed as a **Grid of Blocks of Threads**.

### Visual Example
```
Grid
└── Block 0
    ├── Thread 0
    ├── Thread 1
    └── ...
└── Block 1
    ├── Thread 0
    ├── Thread 1
    └── ...
```

## 4 Key Terms
- `gridDim`: Number of blocks in the grid
- `blockIdx`: Index of the block in the grid
- `blockDim`: Number of threads in a block
- `threadIdx`: Index of the thread in the block

These are built-in variables in CUDA and help each thread know its place.

---

## Threads (Simple)
- Each thread is like a tiny worker with its own private workspace (registers, local memory).
- Example: If you want to add two arrays `a = [1,2,3]` and `b = [2,4,6]`,
  - Thread 0 does `a[0] + b[0]`
  - Thread 1 does `a[1] + b[1]`
  - Thread 2 does `a[2] + b[2]`
- This lets you process lots of data in parallel!

---

## Warps (Simple)
- A **warp** is a group of 32 threads inside a block that execute together.
- The GPU issues instructions to warps, not individual threads.
- The **warp scheduler** decides which warps run at any time.
- You can't avoid using warps—they are how the GPU works under the hood.

### Example
- If your block has 64 threads, it will have 2 warps (since 64 / 32 = 2).
- If your block has 40 threads, it will have 2 warps (since 40 / 32 = 1.25, rounded up to 2).

### Formula
- Number of warps per block = `ceil(T / Wsize)`
  - `T` = number of threads per block
  - `Wsize` = warp size (always 32)
  - `ceil(x)` = round up to the next whole number

#### Example Calculations
- 100 threads per block: `ceil(100 / 32) = 4 warps`
- 32 threads per block: `ceil(32 / 32) = 1 warp`
- 33 threads per block: `ceil(33 / 32) = 2 warps`

---

## Grids in CUDA (Simple Explanation)
- During kernel execution, threads (inside blocks, inside the grid) can access global memory (VRAM).
- A **grid** contains many blocks. Each block contains many threads.
- Grids are great for batch processing: each block in the grid can handle a separate batch element (like processing multiple images at once).

### Why Not Just Use Threads?
- If you only had threads (no blocks), you couldn't use **shared memory**.
- Shared memory is a fast, small memory chunk that is shared among threads in the same block.
- Threads in a block can communicate and cooperate using shared memory, but threads in different blocks cannot.
- Warps (groups of 32 threads) execute together inside blocks, making memory access and communication efficient.

### Example
- Imagine you want to process 1000 images in parallel:
  - You launch a grid with 1000 blocks, each block processes one image.
  - Each block has many threads to process pixels in its image.

### How Blocks Run
- CUDA is scalable because blocks run independently—there are no required order or dependencies between blocks.
- The GPU can run blocks in any order (Block 3 & 0, then Block 6 & 1, etc.).
- Each block solves a piece of the problem, and at the end, all results are combined.

---
**In summary:**
- Grids organize blocks, blocks organize threads.
- Shared memory lets threads in a block work together.
- Blocks run independently, making CUDA scalable and efficient for big problems!

---