# 🧠 CUDA Thread & Block Indexing

This README explains the following CUDA code **in a very beginner‑friendly way**:

```c
#include <stdio.h>

__global__ void whoami(void) {
    int block_id =
        blockIdx.x +
        blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;

    int block_offset =
        block_id *
        blockDim.x * blockDim.y * blockDim.z;

    int thread_offset =
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset;

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main() {
    dim3 blocksPerGrid(2, 3, 4);
    dim3 threadsPerBlock(4, 4, 4);

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}
```

---

## ✅ Compile & Run

Run these commands **from this folder** (`3_First_Kernels/01_CUDA_Basics`).

### Linux / WSL (creates a Linux binary)

```bash
nvcc -o indexing indexing.cu
```
```
./indexing
```

### Windows (PowerShell / Git Bash)

```bash
nvcc -o indexing.exe indexing.cu
./indexing.exe
```

If you see `nvcc: command not found`, install the **NVIDIA CUDA Toolkit** and open a new terminal so your `PATH` updates.

---

## 🚀 What This Program Does

This program does **NOT solve a real problem**.

Its only goal is to **teach how CUDA threads and blocks are organized and indexed**.

Every CUDA thread prints:

* Which **block** it belongs to
* Which **thread** it is inside that block
* Its **global thread ID** across the entire GPU

---

## 🧱 CUDA Execution Model (Simple View)

CUDA runs code using this hierarchy:

```
Grid
 └── Blocks
      └── Threads
```

* **Grid** → Entire kernel launch
* **Block** → Group of threads
* **Thread** → Runs the kernel code

---

## 📦 How Many Blocks Are There?

```c
dim3 blocksPerGrid(2, 3, 4);
```

This means:

| Direction | Blocks |
| --------- | ------ |
| X         | 2      |
| Y         | 3      |
| Z         | 4      |

Total blocks:

```
2 × 3 × 4 = 24 blocks
```

👉 **CUDA creates exactly 24 blocks**.

---

## 👥 How Many Threads Per Block?

```c
dim3 threadsPerBlock(4, 4, 4);
```

Threads per block:

```
4 × 4 × 4 = 64 threads
```

Each block contains **64 threads**.

---

## 🔢 Total Threads Launched

```
24 blocks × 64 threads = 1536 threads
```

So **1536 copies of `whoami()` run in parallel**.

---

## 🧠 Built‑In CUDA Variables

CUDA automatically provides these values to every thread:

| Variable          | Meaning                          |
| ----------------- | -------------------------------- |
| `blockIdx.x/y/z`  | Block position in the grid       |
| `threadIdx.x/y/z` | Thread position inside the block |
| `gridDim.x/y/z`   | Total blocks in each direction   |
| `blockDim.x/y/z`  | Threads per block                |

---

## 🧱 Converting Block Index (3D → 1D)

```c
int block_id =
    blockIdx.x +
    blockIdx.y * gridDim.x +
    blockIdx.z * gridDim.x * gridDim.y;
```

Blocks are **3D**, but we want **one unique block number**.

This formula flattens:

```
(blockIdx.x, blockIdx.y, blockIdx.z)
```

into a **single block ID**.

Example:

```
blockIdx = (1, 1, 0)
gridDim  = (2, 3, 4)

block_id = 1 + 1×2 + 0×6 = 3
```

---

## 👥 Threads Before This Block

```c
int block_offset =
    block_id * blockDim.x * blockDim.y * blockDim.z;
```

Each block has **64 threads**.

If:

```
block_id = 3
```

Then:

```
block_offset = 3 × 64 = 192
```

👉 **192 threads ran before this block**.

---

## 👤 Thread Number Inside the Block

```c
int thread_offset =
    threadIdx.x +
    threadIdx.y * blockDim.x +
    threadIdx.z * blockDim.x * blockDim.y;
```

This converts a **3D thread index** into **one number** inside the block.

Example:

```
threadIdx = (2, 1, 0)
blockDim = (4, 4, 4)

thread_offset = 2 + 1×4 + 0×16 = 6
```

---

## 🌍 Global Thread ID (Most Important Part)

```c
int id = block_offset + thread_offset;
```

This gives **one unique ID** for every thread in the entire grid.

Example:

```
block_offset = 192
thread_offset = 6

Global ID = 198
```

---

## 🖨️ Output Meaning

Example output:

```
0198 | Block(1 1 0) = 3 | Thread(2 1 0) = 6
```

Meaning:

* Global thread ID = 198
* Block position = (1, 1, 0)
* Block number = 3
* Thread position inside block = (2, 1, 0)
* Thread number inside block = 6

---

## ❓ Why X, Y, Z Dimensions?

CUDA supports X, Y, Z because data can be:

| Data Type         | Dimensions |
| ----------------- | ---------- |
| Array             | X          |
| Image / Matrix    | X, Y       |
| 3D Volume / Video | X, Y, Z    |

You **do not have to use all dimensions**.

---

## 🔑 Key Takeaways

* You **decide** how many blocks and threads exist
* CUDA does **not guess** your problem size
* X, Y, Z help match data shape
* Global thread ID = block offset + thread offset

---

## 🧠 One‑Line Summary

> CUDA converts 3D blocks and 3D threads into a single unique global thread ID so every thread knows **who it is**.

---

✅ This code is a **learning demo**, not a real application.
