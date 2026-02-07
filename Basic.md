# CUDA Built-in Variables: `blockIdx`, `blockDim`, `gridDim`, `threadIdx`

Before diving in, let's understand how CUDA organizes work on the GPU.

---

## The Big Picture: Grid → Blocks → Threads

When you launch a CUDA kernel, you create a **grid** of **blocks**, and each block contains **threads**.

```
Grid  (the whole job)
 ├── Block 0
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── ...
 ├── Block 1
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── ...
 └── ...
```

Think of it like a **school**:
- The **Grid** = the entire school
- A **Block** = one classroom
- A **Thread** = one student

Each student (thread) needs to know:
1. **Which classroom am I in?** → `blockIdx`
2. **What is my seat number inside this classroom?** → `threadIdx`
3. **How many classrooms are there in total?** → `gridDim`
4. **How many seats are in each classroom?** → `blockDim`

---

## How You Launch a Kernel

```cpp
// kernel<<<number_of_blocks, threads_per_block>>>();
myKernel<<<4, 8>>>();
```

This launches **4 blocks**, each with **8 threads** = **32 threads total**.

You can also use `dim3` for 2D or 3D layouts:

```cpp
dim3 blocks(2, 3);      // 2×3 = 6 blocks
dim3 threads(4, 4);      // 4×4 = 16 threads per block
myKernel<<<blocks, threads>>>();
// Total: 6 × 16 = 96 threads
```

Now let's look at each built-in variable one by one.

---

## 1. `gridDim` — How many blocks are in the grid?

`gridDim` tells every thread the **size (shape) of the grid**, i.e., how many blocks exist in each direction.

### Components
| Component    | Meaning                        |
|-------------|--------------------------------|
| `gridDim.x` | Number of blocks in X direction |
| `gridDim.y` | Number of blocks in Y direction |
| `gridDim.z` | Number of blocks in Z direction |

### Analogy
> **"How many classrooms does the school have in each row/column/floor?"**

### Example 1 — Simple 1D grid

```cpp
myKernel<<<4, 256>>>();
```

Inside the kernel:
```
gridDim.x = 4      // 4 blocks in X
gridDim.y = 1      // default
gridDim.z = 1      // default
```

### Example 2 — 2D grid

```cpp
dim3 grid(3, 2);          // 3 columns, 2 rows of blocks
myKernel<<<grid, 128>>>();
```

```
gridDim.x = 3
gridDim.y = 2
gridDim.z = 1
```

Visual layout of the 6 blocks:

```
         col 0      col 1      col 2
       ┌──────────┬──────────┬──────────┐
row 0  │ Block    │ Block    │ Block    │
       │ (0,0)    │ (1,0)    │ (2,0)    │
       ├──────────┼──────────┼──────────┤
row 1  │ Block    │ Block    │ Block    │
       │ (0,1)    │ (1,1)    │ (2,1)    │
       └──────────┴──────────┴──────────┘

gridDim.x = 3 (number of columns)
gridDim.y = 2 (number of rows)
```

### Key Point
> `gridDim` is the **same value for every thread** in the entire kernel launch. It's a **constant** — it describes the shape, not a position.

---

## 2. `blockIdx` — Which block am I in?

`blockIdx` tells each thread **where its block is located** inside the grid.

### Components
| Component     | Meaning                          |
|--------------|----------------------------------|
| `blockIdx.x` | This block's position along X    |
| `blockIdx.y` | This block's position along Y    |
| `blockIdx.z` | This block's position along Z    |

### Analogy
> **"What is my classroom number?"**

### Example — 1D grid with 4 blocks

```cpp
myKernel<<<4, 256>>>();
```

Each block gets a different `blockIdx.x`:

```
Block 0 → blockIdx.x = 0
Block 1 → blockIdx.x = 1
Block 2 → blockIdx.x = 2
Block 3 → blockIdx.x = 3
```

### Example — 2D grid (3×2)

```cpp
dim3 grid(3, 2);
myKernel<<<grid, 128>>>();
```

```
Block at (0,0) → blockIdx.x = 0, blockIdx.y = 0
Block at (1,0) → blockIdx.x = 1, blockIdx.y = 0
Block at (2,0) → blockIdx.x = 2, blockIdx.y = 0
Block at (0,1) → blockIdx.x = 0, blockIdx.y = 1
Block at (1,1) → blockIdx.x = 1, blockIdx.y = 1
Block at (2,1) → blockIdx.x = 2, blockIdx.y = 1
```

### Key Point
> `blockIdx` is **different for each block**, but **the same for all threads within the same block**. Every student in Classroom 5 reports `blockIdx = 5`.

---

## 3. `blockDim` — How many threads are in one block?

`blockDim` tells every thread **the size (shape) of its block**, i.e., how many threads exist in each direction within a single block.

### Components
| Component     | Meaning                              |
|--------------|--------------------------------------|
| `blockDim.x` | Number of threads in X direction     |
| `blockDim.y` | Number of threads in Y direction     |
| `blockDim.z` | Number of threads in Z direction     |

### Analogy
> **"How many seats are in each row/column of my classroom?"**

### Example 1 — Simple 1D block

```cpp
myKernel<<<4, 256>>>();
//              ^^^  256 threads per block
```

Inside the kernel:
```
blockDim.x = 256
blockDim.y = 1
blockDim.z = 1
```

### Example 2 — 2D block

```cpp
dim3 threads(16, 16);          // 16×16 = 256 threads per block
myKernel<<<4, threads>>>();
```

```
blockDim.x = 16
blockDim.y = 16
blockDim.z = 1
```

Visual layout of threads inside ONE block:

```
          threadIdx.x →
          0    1    2    3   ...  15
       ┌────┬────┬────┬────┬───┬────┐
  0    │ T  │ T  │ T  │ T  │...│ T  │
       ├────┼────┼────┼────┼───┼────┤
  1    │ T  │ T  │ T  │ T  │...│ T  │
       ├────┼────┼────┼────┼───┼────┤
  ...  │    │    │    │    │   │    │
       ├────┼────┼────┼────┼───┼────┤
  15   │ T  │ T  │ T  │ T  │...│ T  │
       └────┴────┴────┴────┴───┴────┘
  ↑
  threadIdx.y

blockDim.x = 16 (columns of threads)
blockDim.y = 16 (rows of threads)
```

### Key Point
> `blockDim` is the **same value for every thread** in the entire kernel. It's a **constant** — it tells you the shape of each block, not a position.

---

## 4. `threadIdx` — Which thread am I inside my block?

`threadIdx` tells each thread **its position inside its own block**.

### Components
| Component      | Meaning                                  |
|---------------|------------------------------------------|
| `threadIdx.x` | This thread's position along X in block  |
| `threadIdx.y` | This thread's position along Y in block  |
| `threadIdx.z` | This thread's position along Z in block  |

### Analogy
> **"What is my seat number inside my classroom?"**

### Example — 1D block with 8 threads

```cpp
myKernel<<<2, 8>>>();
```

Inside Block 0:
```
Thread 0 → threadIdx.x = 0
Thread 1 → threadIdx.x = 1
Thread 2 → threadIdx.x = 2
...
Thread 7 → threadIdx.x = 7
```

Inside Block 1: **the exact same thing!**
```
Thread 0 → threadIdx.x = 0
Thread 1 → threadIdx.x = 1
...
Thread 7 → threadIdx.x = 7
```

### Key Point
> `threadIdx` **resets to 0 in every block**. Two threads in different blocks can have the same `threadIdx`. It's like seat numbers restarting at 1 in every classroom.

---

## Summary Table

| Variable     | What it tells you                     | Changes per... | Analogy                          |
|-------------|---------------------------------------|----------------|----------------------------------|
| `gridDim`   | Total number of blocks in each axis   | Never (constant) | Number of classrooms in school |
| `blockIdx`  | This block's position in the grid     | Block          | My classroom number              |
| `blockDim`  | Number of threads per block (each axis)| Never (constant) | Seats per classroom            |
| `threadIdx` | This thread's position in its block   | Thread         | My seat number                   |

---

## Putting It All Together: Computing a Global Thread ID

The most common use case: you have an array of N elements and want each thread to process one element. You need a **unique global ID** for each thread.

### 1D Case (most common)

```cpp
__global__ void addArrays(float *a, float *b, float *c, int N) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    //             ^^^^^^^^^^   ^^^^^^^^^^   ^^^^^^^^^^^
    //             which block   block size   seat in block

    if (globalId < N) {
        c[globalId] = a[globalId] + b[globalId];
    }
}
```

**How it works step by step:**

```cpp
myKernel<<<3, 4>>>();   // 3 blocks, 4 threads each = 12 threads
```

```
Block 0 (blockIdx.x = 0):
  Thread 0: globalId = 0 * 4 + 0 = 0
  Thread 1: globalId = 0 * 4 + 1 = 1
  Thread 2: globalId = 0 * 4 + 2 = 2
  Thread 3: globalId = 0 * 4 + 3 = 3

Block 1 (blockIdx.x = 1):
  Thread 0: globalId = 1 * 4 + 0 = 4
  Thread 1: globalId = 1 * 4 + 1 = 5
  Thread 2: globalId = 1 * 4 + 2 = 6
  Thread 3: globalId = 1 * 4 + 3 = 7

Block 2 (blockIdx.x = 2):
  Thread 0: globalId = 2 * 4 + 0 = 8
  Thread 1: globalId = 2 * 4 + 1 = 9
  Thread 2: globalId = 2 * 4 + 2 = 10
  Thread 3: globalId = 2 * 4 + 3 = 11
```

Every thread gets a unique ID from 0 to 11!

### 2D Case (useful for images/matrices)

```cpp
__global__ void processImage(unsigned char *img, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // X position
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Y position

    if (col < width && row < height) {
        int pixel = row * width + col;
        img[pixel] = 255 - img[pixel];  // invert color
    }
}
```

### 3D Case (code from indexing.cu)

```cpp
// Flatten 3D block index to 1D
int block_id = blockIdx.x
             + blockIdx.y * gridDim.x
             + blockIdx.z * gridDim.x * gridDim.y;

// How many threads exist before this block
int block_offset = block_id * (blockDim.x * blockDim.y * blockDim.z);

// Flatten 3D thread index to 1D within the block
int thread_offset = threadIdx.x
                  + threadIdx.y * blockDim.x
                  + threadIdx.z * blockDim.x * blockDim.y;

// Final unique global ID
int globalId = block_offset + thread_offset;
```

---

## Common Mistakes

### 1. Forgetting bounds checking
```cpp
// BAD — accesses out-of-bounds memory if N isn't a multiple of blockDim
__global__ void bad(float *a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = 0;  // CRASH if i >= N
}

// GOOD
__global__ void good(float *a, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {         // always check!
        a[i] = 0;
    }
}
```

### 2. Confusing `blockDim` with `gridDim`
- `blockDim` = threads per block (size of a block)
- `gridDim` = blocks per grid (size of the grid)

### 3. Thinking `threadIdx` is globally unique
`threadIdx` only identifies a thread **within its block**. Thread 0 exists in every block. To get a globally unique ID, combine it with `blockIdx` and `blockDim`.

---

## Quick Reference

```
gridDim.x      → number of blocks in X
gridDim.y      → number of blocks in Y
gridDim.z      → number of blocks in Z

blockIdx.x     → this block's X position   (0 to gridDim.x - 1)
blockIdx.y     → this block's Y position   (0 to gridDim.y - 1)
blockIdx.z     → this block's Z position   (0 to gridDim.z - 1)

blockDim.x     → threads per block in X
blockDim.y     → threads per block in Y
blockDim.z     → threads per block in Z

threadIdx.x    → this thread's X position  (0 to blockDim.x - 1)
threadIdx.y    → this thread's Y position  (0 to blockDim.y - 1)
threadIdx.z    → this thread's Z position  (0 to blockDim.z - 1)

Global ID (1D) = blockIdx.x * blockDim.x + threadIdx.x
```
