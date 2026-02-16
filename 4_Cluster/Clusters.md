# CUDA Thread Block Clusters -- Complete Guide

> **Note**: Thread Block Clusters require Compute Capability 9.0+ (H100 GPUs and newer)

---

## 📚 Table of Contents
1. [Traditional CUDA Hierarchy](#1-traditional-cuda-hierarchy-before-clusters)
2. [Why Clusters Were Introduced](#2-why-clusters-were-introduced)
3. [Method 1: Using __cluster_dims__](#3-method-1-using-__cluster_dims__-implicit-clusters)
4. [Method 2: Using __block_size__](#4-method-2-using-__block_size__-explicit-clusters)
5. [Comparing Both Methods](#5-comparing-both-methods)
6. [Practical Code Examples](#6-practical-code-examples)
7. [Real-World Use Cases](#7-real-world-use-cases)

---

## 1. Traditional CUDA Hierarchy (Before Clusters)

### 📊 The Original 3-Level Hierarchy

```
Grid
 ├── Block 0
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── Thread ...
 ├── Block 1
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── Thread ...
 └── Block ...
```

### Basic Kernel Launch Syntax

```cpp
kernelName<<<gridDim, blockDim>>>();
```

- **`gridDim`** = Number of blocks in the grid
- **`blockDim`** = Number of threads per block

### 💡 Example 1: Simple Launch

```cpp
// Launch 256 blocks, each with 128 threads
myKernel<<<256, 128>>>();
```

**Calculations:**
- Total blocks = 256
- Threads per block = 128
- **Total threads = 256 × 128 = 32,768 threads**

### 💡 Example 2: 3D Launch

```cpp
// Launch in 3D configuration
myKernel<<<dim3(16, 16, 16), dim3(32, 32, 1)>>>();
```

**Calculations:**
- Total blocks = 16 × 16 × 16 = **4,096 blocks**
- Threads per block = 32 × 32 × 1 = **1,024 threads/block**
- **Total threads = 4,096 × 1,024 = 4,194,304 threads**

### ⚠️ Limitations of Traditional Hierarchy

```cpp
__global__ void traditionalKernel() {
    __shared__ int sharedData[256];
    
    // ✅ Threads in SAME block can synchronize
    __syncthreads();
    
    // ❌ Threads in DIFFERENT blocks CANNOT synchronize
    // No inter-block communication!
    // Each block is completely independent
}
```

---

## 2. Why Clusters Were Introduced

### 🚫 Problems with Traditional Blocks

| Feature | Traditional Blocks | With Clusters |
|---------|-------------------|---------------|
| **Inter-block sync** | ❌ Impossible | ✅ Possible |
| **Shared memory** | Only within block | Across blocks in cluster |
| **Scheduling guarantee** | Independent | Co-scheduled on same GPC |
| **Communication** | Via global memory | Via distributed shared memory |

### 🎯 What Clusters Enable

#### Before Clusters (CC < 9.0):
```cpp
__global__ void oldWay() {
    __shared__ int blockData[256];
    
    // ✅ Can sync with threads in MY block
    __syncthreads();
    
    // ❌ Cannot sync with OTHER blocks
    // If I need data from another block:
    // 1. Write to global memory
    // 2. Exit kernel
    // 3. Launch new kernel
    // 4. Read from global memory
    // Very slow! 😢
}
```

#### With Clusters (CC >= 9.0):
```cpp
__cluster_dims__(2, 2, 1)  // Cluster of 4 blocks
__global__ void newWay() {
    __shared__ int blockData[256];
    
    // ✅ Can sync with threads in MY block
    __syncthreads();
    
    // ✅ Can sync with OTHER blocks in MY cluster
    cluster.sync();
    
    // ✅ Can access shared memory from other blocks in cluster
    // Much faster! 😊
}
```

### 📊 New 4-Level Hierarchy

```
Grid
 ├── Cluster 0
 │    ├── Block 0
 │    │    ├── Thread 0
 │    │    ├── Thread 1
 │    │    └── Thread ...
 │    ├── Block 1
 │    │    └── Threads...
 │    └── Block ...
 ├── Cluster 1
 │    ├── Block 0
 │    ├── Block 1
 │    └── Block ...
 └── Cluster ...
```

**Key**: Blocks within the **same cluster** can cooperate!

---

## 3. Method 1: Using `__cluster_dims__` (Implicit Clusters)

### 🎯 Concept: You Launch Blocks, CUDA Groups Them

With `__cluster_dims__`, you specify the **cluster shape**, then launch blocks as usual. CUDA automatically groups your blocks into clusters.

### Basic Syntax

```cpp
__cluster_dims__(X, Y, Z)  // X×Y×Z blocks per cluster
__global__ void myKernel() {
    // Kernel code
}

// Launch: You specify NUMBER OF BLOCKS
myKernel<<<numBlocks, threadsPerBlock>>>();
```

---

### 💡 Example 1: Simple 1D Cluster

```cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Each cluster contains 4 blocks (4×1×1)
__cluster_dims__(4, 1, 1)
__global__ void clusterKernel() {
    namespace cg = cooperative_groups;
    
    // Get cluster information
    cg::cluster_group cluster = cg::this_cluster();
    
    int blockIdxInCluster = cluster.block_rank();  // 0 to 3
    int clusterSize = cluster.num_blocks();         // 4
    
    if (threadIdx.x == 0) {
        printf("Block %d is block #%d in its cluster of %d blocks\n",
               blockIdx.x, blockIdxInCluster, clusterSize);
    }
    
    // Synchronize all blocks in this cluster
    cluster.sync();
}

int main() {
    // Launch 16 blocks, each with 256 threads
    clusterKernel<<<16, 256>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
```

**📊 What Happens:**
```
Total blocks launched = 16
Blocks per cluster = 4
Number of clusters = 16 ÷ 4 = 4 clusters

Cluster 0: [Block 0] [Block 1] [Block 2] [Block 3]
Cluster 1: [Block 4] [Block 5] [Block 6] [Block 7]
Cluster 2: [Block 8] [Block 9] [Block 10] [Block 11]
Cluster 3: [Block 12] [Block 13] [Block 14] [Block 15]
```

---

### 💡 Example 2: 2D Cluster (Most Common)

```cpp
// Each cluster contains 2×2 = 4 blocks
__cluster_dims__(2, 2, 1)
__global__ void matrixKernel(float* data) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    
    // Cluster dimensions
    dim3 clusterIdx = cluster.dim_blocks();  // (2, 2, 1)
    dim3 blockInCluster = cluster.block_rank();
    
    printf("Block (%d,%d) is at position (%d,%d) in cluster\n",
           blockIdx.x, blockIdx.y, 
           blockInCluster.x, blockInCluster.y);
}

int main() {
    // Launch 8×8 = 64 blocks
    matrixKernel<<<dim3(8, 8, 1), dim3(16, 16, 1)>>>(data);
}
```

**📊 What Happens:**
```
Total blocks = 8 × 8 = 64 blocks
Blocks per cluster = 2 × 2 = 4 blocks
Number of clusters = 64 ÷ 4 = 16 clusters
Cluster grid = 4 × 4 clusters

        Cluster Grid (4×4)
     0      1      2      3
  ┌─────┬─────┬─────┬─────┐
0 │ C0  │ C1  │ C2  │ C3  │
  ├─────┼─────┼─────┼─────┤
1 │ C4  │ C5  │ C6  │ C7  │
  ├─────┼─────┼─────┼─────┤
2 │ C8  │ C9  │ C10 │ C11 │
  ├─────┼─────┼─────┼─────┤
3 │ C12 │ C13 │ C14 │ C15 │
  └─────┴─────┴─────┴─────┘

Each cluster contains 2×2 blocks:
C0 = {Block(0,0), Block(0,1), Block(1,0), Block(1,1)}
```

---

### 💡 Example 3: 3D Cluster

```cpp
// Each cluster contains 2×2×2 = 8 blocks
__cluster_dims__(2, 2, 2)
__global__ void volume3DKernel() {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    
    int blocksPerCluster = cluster.num_blocks();  // 8
}

int main() {
    // Launch 16×16×16 = 4096 blocks
    volume3DKernel<<<dim3(16, 16, 16), dim3(1024, 1, 1)>>>();
}
```

**📊 Calculations:**
```
Total blocks = 16 × 16 × 16 = 4,096 blocks
Blocks per cluster = 2 × 2 × 2 = 8 blocks
Number of clusters = 4,096 ÷ 8 = 512 clusters
Cluster grid dimensions = 8 × 8 × 8
```

---

### 🔑 Key Points for `__cluster_dims__`

✅ **You launch**: Number of **blocks**  
✅ **CUDA groups**: Blocks into **clusters**  
✅ **You calculate**: Number of clusters = Total blocks ÷ Blocks per cluster  
✅ **blockIdx**: Still refers to global block index (0 to numBlocks-1)  
✅ **cluster.block_rank()**: Local block index within cluster (0 to clusterSize-1)

---

## 4. Method 2: Using `__block_size__` (Explicit Clusters)

### 🎯 Concept: You Launch Clusters Directly

With `__block_size__`, you define **both** thread dimensions **and** cluster dimensions together. Then you launch **clusters** (not blocks).

### Basic Syntax

```cpp
__block_size__((threadsX, threadsY, threadsZ), (clustX, clustY, clustZ))
__global__ void myKernel() {
    // Kernel code
}

// Launch: You specify NUMBER OF CLUSTERS
myKernel<<<numClusters>>>();
```

**⚠️ IMPORTANT**: The first argument in `<<<>>>` now means **number of clusters**, NOT number of blocks!

---

### 💡 Example 1: Simple Configuration

```cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// 1024 threads/block, 4 blocks/cluster
__block_size__((1024, 1, 1), (4, 1, 1))
__global__ void clusterKernel() {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Cluster has %d blocks\n", cluster.num_blocks());
    }
}

int main() {
    // Launch 16 CLUSTERS (not blocks!)
    clusterKernel<<<16>>>();
    // Each cluster has 4 blocks
    // Total blocks = 16 × 4 = 64 blocks
    
    cudaDeviceSynchronize();
    return 0;
}
```

**📊 What Happens:**
```
YOU LAUNCHED: 16 clusters
Each cluster has: 4 blocks
Total blocks = 16 × 4 = 64 blocks
Each block has: 1,024 threads
Total threads = 64 × 1,024 = 65,536 threads
```

---

### 💡 Example 2: 2D Cluster Launch

```cpp
// 32×32 threads per block, 2×2 blocks per cluster
__block_size__((32, 32, 1), (2, 2, 1))
__global__ void matrixKernel(float* A, float* B) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Process data...
    
    // Sync all blocks in cluster
    cluster.sync();
}

int main() {
    // Launch 8×8 = 64 CLUSTERS
    matrixKernel<<<dim3(8, 8, 1)>>>(A, B);
}
```

**📊 Detailed Breakdown:**
```
YOU LAUNCHED: 8 × 8 = 64 clusters

Each cluster contains: 2 × 2 = 4 blocks
Total blocks = 64 clusters × 4 blocks/cluster = 256 blocks
Block grid = 16 × 16 (because 8 clusters/dim × 2 blocks/cluster = 16)

Each block contains: 32 × 32 = 1,024 threads
Total threads = 256 blocks × 1,024 threads/block = 262,144 threads

Visual (cluster-level):
   Cluster Grid (8×8)
   Each [C] contains 4 blocks

   [C0] [C1] [C2] ... [C7]
   [C8] [C9] ...      [C15]
   ...
```

---

### 💡 Example 3: Same Result, Different Methods

Here's the **same execution** using both methods:

#### Method A: Using `__cluster_dims__`
```cpp
__cluster_dims__(2, 2, 1)
__global__ void kernelA() { }

int main() {
    // Launch 16×16 = 256 BLOCKS
    kernelA<<<dim3(16, 16, 1), dim3(32, 32, 1)>>>();
    
    // Results:
    // - 256 blocks total
    // - 4 blocks per cluster (2×2)
    // - 64 clusters (256÷4)
}
```

#### Method B: Using `__block_size__`
```cpp
__block_size__((32, 32, 1), (2, 2, 1))
__global__ void kernelB() { }

int main() {
    // Launch 8×8 = 64 CLUSTERS
    kernelB<<<dim3(8, 8, 1)>>>();
    
    // Results:
    // - 64 clusters
    // - 4 blocks per cluster (2×2)
    // - 256 blocks total (64×4)
}
```

**Both produce identical execution!** The difference is conceptual:
- Method A: Think in **blocks**, CUDA organizes clusters
- Method B: Think in **clusters** directly

---

### 🔑 Key Points for `__block_size__`

✅ **You launch**: Number of **clusters** (not blocks!)  
✅ **Block dimensions**: Defined in first tuple  
✅ **Cluster dimensions**: Defined in second tuple  
✅ **Total blocks**: numClusters × blocksPerCluster  
✅ **Clearer**: Explicitly shows you're working with clusters

---

## 5. Comparing Both Methods

### 📊 Side-by-Side Comparison

| Feature | `__cluster_dims__` | `__block_size__` |
|---------|-------------------|------------------|
| **What you launch** | Number of blocks | Number of clusters |
| **Cluster size** | Specified separately | Specified with block size |
| **Launch syntax** | `<<<blocks, threads>>>` | `<<<clusters>>>` |
| **Thread dimensions** | In launch call | In attribute |
| **Mental model** | Blocks that CUDA groups | Clusters directly |
| **Flexibility** | Can vary threads at launch | Threads fixed at compile time |
| **Clarity** | Need to calculate clusters | Clusters explicit |

### 🔄 Translation Table

To achieve the same execution:

| `__cluster_dims__` Version | `__block_size__` Version |
|----------------------------|--------------------------|
| `__cluster_dims__(4, 1, 1)` | `__block_size__((threads...), (4, 1, 1))` |
| Launch `<<<64, 256>>>` | Launch `<<<16>>>` |
| → 64 blocks, 4 per cluster | → 16 clusters, 4 blocks each |
| → **16 clusters total** | → **16 clusters total** |

### 💡 When to Use Which?

#### Use `__cluster_dims__` when:
- ✅ You want flexibility in thread count at launch time
- ✅ You're used to thinking in "blocks"
- ✅ You're converting existing non-cluster code
- ✅ Thread count varies between kernel launches

#### Use `__block_size__` when:
- ✅ You want to think directly in terms of clusters
- ✅ Thread and cluster dimensions are fixed
- ✅ You want clearer, more explicit code
- ✅ You're writing new cluster-aware code

### ⚠️ Important Restrictions

```cpp
// ❌ ILLEGAL: Cannot use both
__cluster_dims__(2, 2, 1)
__block_size__((256, 1, 1), (2, 2, 1))  // ERROR!
__global__ void badKernel() { }

// ✅ LEGAL: Choose one method
__cluster_dims__(2, 2, 1)
__global__ void goodKernel1() { }

// ✅ LEGAL: Or the other
__block_size__((256, 1, 1), (2, 2, 1))
__global__ void goodKernel2() { }
```

**Why?** Both methods define cluster size - using both would be ambiguous and conflicting.

---

## 6. Practical Code Examples

### 💡 Example 1: Distributed Shared Memory

Blocks in a cluster can access shared memory from other blocks:

```cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__cluster_dims__(2, 1, 1)  // 2 blocks per cluster
__global__ void distributedSharedMemoryKernel(int* output) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    
    // Regular shared memory for this block
    __shared__ int localData[256];
    
    // Initialize local data
    localData[threadIdx.x] = blockIdx.x * 1000 + threadIdx.x;
    __syncthreads();
    
    // Synchronize entire cluster
    cluster.sync();
    
    // Access shared memory from OTHER blocks in cluster
    if (threadIdx.x == 0) {
        int myBlockRank = cluster.block_rank();  // 0 or 1
        int otherBlockRank = 1 - myBlockRank;     // 1 or 0
        
        // Get address of other block's shared memory
        int* otherBlockData = cluster.map_shared_rank(localData, otherBlockRank);
        
        // Read from partner block's shared memory!
        printf("Block %d reads from Block %d: value=%d\n",
               myBlockRank, otherBlockRank, otherBlockData[0]);
    }
}

int main() {
    // Launch 4 blocks → 2 clusters of 2 blocks each
    distributedSharedMemoryKernel<<<4, 256>>>(nullptr);
    cudaDeviceSynchronize();
    return 0;
}
```

**Output:**
```
Block 0 reads from Block 1: value=1000
Block 1 reads from Block 0: value=0
Block 0 reads from Block 1: value=3000
Block 1 reads from Block 0: value=2000
```

---

### 💡 Example 2: Cluster-Wide Reduction

Perform reduction across multiple blocks efficiently:

```cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__cluster_dims__(4, 1, 1)  // 4 blocks per cluster
__global__ void clusterReduction(int* input, int* output, int N) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    
    __shared__ int blockSum;
    
    // Step 1: Each thread processes elements
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int value = (globalIdx < N) ? input[globalIdx] : 0;
    
    // Step 2: Block-level reduction
    __shared__ int sharedData[256];
    sharedData[threadIdx.x] = value;
    __syncthreads();
    
    // Simple reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        blockSum = sharedData[0];
    }
    __syncthreads();
    
    // Step 3: Cluster-level reduction
    cluster.sync();  // All blocks in cluster ready
    
    // First block in cluster collects results from all blocks
    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        int clusterSum = blockSum;  // Start with own block
        
        // Add sums from other blocks in cluster
        for (int i = 1; i < cluster.num_blocks(); i++) {
            int* otherBlockSum = cluster.map_shared_rank(&blockSum, i);
            clusterSum += *otherBlockSum;
        }
        
        // Write cluster result
        int clusterIdx = blockIdx.x / cluster.num_blocks();
        output[clusterIdx] = clusterSum;
    }
}

int main() {
    const int N = 4096;
    int* d_input, *d_output;
    
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, 256 * sizeof(int));
    
    // Launch 16 blocks → 4 clusters
    clusterReduction<<<16, 256>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
```

---

### 💡 Example 3: Matrix Multiplication with Clusters

```cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define TILE_SIZE 16

__cluster_dims__(2, 2, 1)  // 2×2 block cluster
__global__ void matMulCluster(float* A, float* B, float* C, int N) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Load tiles and compute
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = 
                A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B
        if (col < N && t * TILE_SIZE + threadIdx.y < N) {
            tileB[threadIdx.y][threadIdx.x] = 
                B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Cluster sync before writing results
    cluster.sync();
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 512;
    float *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));
    
    // Launch grid: 32×32 blocks → 16×16 clusters
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (N + TILE_SIZE - 1) / TILE_SIZE);
    
    matMulCluster<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
```

---

### 💡 Example 4: Querying Cluster Information

```cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

__cluster_dims__(2, 2, 1)
__global__ void queryClusterInfo() {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    
    // Only one thread prints per block
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Cluster information
        int numBlocks = cluster.num_blocks();           // 4
        int blockRank = cluster.block_rank();            // 0-3
        dim3 clusterDims = cluster.dim_blocks();         // (2,2,1)
        
        // Block position within cluster
        dim3 blockPos;
        blockPos.x = blockRank % clusterDims.x;
        blockPos.y = (blockRank / clusterDims.x) % clusterDims.y;
        blockPos.z = blockRank / (clusterDims.x * clusterDims.y);
        
        printf("Global Block (%d,%d,%d) → "
               "Cluster Block Rank %d → "
               "Position in Cluster (%d,%d,%d)\n",
               blockIdx.x, blockIdx.y, blockIdx.z,
               blockRank,
               blockPos.x, blockPos.y, blockPos.z);
    }
}

int main() {
    // Launch 4×4 = 16 blocks → 2×2 = 4 clusters
    queryClusterInfo<<<dim3(4, 4, 1), dim3(8, 8, 1)>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

---

## 7. Real-World Use Cases

### 🎯 When to Use Thread Block Clusters

| Use Case | Why Clusters Help |
|----------|-------------------|
| **Large Reductions** | Reduce across blocks without round-tripping to global memory |
| **Histograms** | Multiple blocks can update shared histogram bins cooperatively |
| **Sparse Matrix Ops** | Coordinate work across irregular data distributions |
| **Graph Algorithms** | Blocks processing neighboring nodes can share edge data |
| **FFT/Signal Processing** | Share intermediate results between processing stages |
| **Ray Tracing** | Blocks can share BVH traversal data |
| **Molecular Dynamics** | Particles in adjacent spatial regions cooperate |

---

### 🏢 Real-World Analogy

#### Without Clusters (Traditional CUDA):
```
🏙️ City (Grid)
├── 🏢 Building 1 (Block)
│   └── 👷 Workers (Threads)
│       ❌ Cannot talk to other buildings
├── 🏢 Building 2 (Block)
│   └── 👷 Workers (Threads)
│       ❌ Cannot talk to other buildings
└── 🏢 Building 3 (Block)
    └── 👷 Workers (Threads)
        ❌ Cannot talk to other buildings

Problem: Buildings are isolated!
- If Building 1 needs data from Building 2, must use slow city-wide mail (global memory)
```

#### With Clusters:
```
🏙️ City (Grid)
├── 🏛️ Department A (Cluster)
│   ├── 🏢 Building 1 (Block)
│   │   └── 👷 Workers (Threads)
│   │       ✅ Can talk within department
│   └── 🏢 Building 2 (Block)
│       └── 👷 Workers (Threads)
│           ✅ Can talk within department
│
└── 🏛️ Department B (Cluster)
    ├── 🏢 Building 3 (Block)
    │   └── 👷 Workers (Threads)
    │       ✅ Can talk within department
    └── 🏢 Building 4 (Block)
        └── 👷 Workers (Threads)
            ✅ Can talk within department

Solution: Buildings in same department share local network!
- Buildings 1 & 2 can share data quickly (distributed shared memory)
- Departments A and B remain independent
```

---

### 📊 Performance Benefits

#### Traditional Approach (Without Clusters):
```cpp
// Kernel 1: Each block processes data
processKernel<<<numBlocks, threadsPerBlock>>>();
cudaDeviceSynchronize();  // ⏱️ Wait for ALL blocks

// Write partial results to global memory
// ⏱️ Slow global memory access

// Kernel 2: Combine results
combineKernel<<<1, numBlocks>>>();
cudaDeviceSynchronize();  // ⏱️ Wait again

// Total: 2 kernel launches + 2 synchronizations + global memory
```

#### With Clusters:
```cpp
// Single kernel: Process AND combine within cluster
__cluster_dims__(8, 1, 1)
__global__ void processAndCombine() {
    // Process data
    __syncthreads();  // Sync within block
    
    // Combine within cluster
    cluster.sync();   // ⚡ Fast cluster sync
    // Share via distributed shared memory
    
    // Done!
}

processAndCombine<<<numBlocks, threadsPerBlock>>>();
cudaDeviceSynchronize();  // ⏱️ Single wait

// Total: 1 kernel launch + 1 synchronization + fast shared memory
```

**Result**: Up to **2-4× faster** for reduction/combining workloads! 🚀

---

### 🔬 Research Applications

1. **Molecular Dynamics Simulation**
```cpp
__cluster_dims__(4, 4, 4)  // 64 blocks per cluster
__global__ void computeForces(Particle* particles) {
    // Each block handles spatial region
    // Clusters group neighboring regions
    // Share particle data across boundaries efficiently
    cluster.sync();
}
```

2. **Deep Learning - Custom Kernels**
```cpp
__cluster_dims__(2, 2, 1)  // 4 blocks per cluster
__global__ void customAttention(float* Q, float* K, float* V) {
    // Split attention heads across blocks
    // Share intermediate QK^T results within cluster
    cluster.sync();
}
```

3. **Computational Fluid Dynamics**
```cpp
__cluster_dims__(2, 2, 2)  // 8 blocks per cluster
__global__ void fluidSolver(float* velocityField) {
    // Each block handles grid cell
    // Exchange boundary conditions within cluster
    cluster.sync();
}
```

---

### ⚡ Quick Reference Card

```cpp
// ═══════════════════════════════════════════════════════════
//  CUDA THREAD BLOCK CLUSTERS - QUICK REFERENCE
// ═══════════════════════════════════════════════════════════

// METHOD 1: __cluster_dims__ (Launch Blocks)
// ───────────────────────────────────────────────────────────
__cluster_dims__(2, 2, 1)                    // 4 blocks/cluster
__global__ void kernel1() { 
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    cluster.sync();                          // Sync cluster
}
kernel1<<<dim3(8,8,1), dim3(16,16,1)>>>();  // Launch 64 BLOCKS
// Result: 64 blocks → 16 clusters

// METHOD 2: __block_size__ (Launch Clusters)
// ───────────────────────────────────────────────────────────
__block_size__((16,16,1), (2,2,1))          // threads + cluster
__global__ void kernel2() { 
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    cluster.sync();                          // Sync cluster
}
kernel2<<<dim3(8,8,1)>>>();                 // Launch 64 CLUSTERS
// Result: 64 clusters → 256 blocks

// CLUSTER API
// ───────────────────────────────────────────────────────────
cluster.sync();                              // Sync all blocks
cluster.num_blocks();                        // Blocks in cluster
cluster.block_rank();                        // My block index (0-N)
cluster.dim_blocks();                        // Cluster dimensions
cluster.map_shared_rank(ptr, rank);          // Access other block's memory

// REQUIREMENTS
// ───────────────────────────────────────────────────────────
// ✅ Compute Capability 9.0+ (H100, H200, etc.)
// ✅ #include <cooperative_groups.h>
// ✅ Compile with: nvcc -arch=sm_90 ...
```

---

## 📚 Summary

### Key Takeaways

1. **Traditional CUDA**: Grid → Blocks → Threads (3 levels)
2. **With Clusters**: Grid → Clusters → Blocks → Threads (4 levels)

3. **Two Methods**:
   - `__cluster_dims__`: Launch **blocks**, CUDA groups them
   - `__block_size__`: Launch **clusters** directly

4. **Benefits**:
   - Inter-block synchronization (`cluster.sync()`)
   - Distributed shared memory access
   - Better performance for cooperative algorithms
   - Reduced global memory traffic

5. **When to Use**:
   - Large reductions
   - Histograms
   - Cooperative algorithms
   - Any time blocks need to communicate

---

### Quick Comparison Table

| Aspect | `__cluster_dims__` | `__block_size__` |
|--------|-------------------|------------------|
| **Syntax** | `__cluster_dims__(2,2,1)` | `__block_size__((threads), (2,2,1))` |
| **Launch meaning** | Number of blocks | Number of clusters |
| **Launch example** | `<<<64, 256>>>` | `<<<16>>>` |
| **Thread flexibility** | ✅ Can change at launch | ❌ Fixed at compile time |
| **Mental model** | Bottom-up (blocks→clusters) | Top-down (clusters→blocks) |
| **Best for** | Converting existing code | New cluster-aware code |

---

### Hardware Requirements

```
┌──────────────────────────────────────────────────────┐
│  CUDA Compute Capability Requirements               │
├──────────────────────────────────────────────────────┤
│  CC < 9.0  │  ❌ No cluster support                 │
│  CC ≥ 9.0  │  ✅ Full cluster support               │
├──────────────────────────────────────────────────────┤
│  Supported GPUs:                                     │
│  • NVIDIA H100 (Hopper)                              │
│  • NVIDIA H200                                       │
│  • Future architectures                              │
└──────────────────────────────────────────────────────┘
```

---

## 🎓 Final Notes

**Remember**:
- With `__cluster_dims__`: You launch **blocks**, think bottom-up
- With `__block_size__`: You launch **clusters**, think top-down
- Both achieve the same execution, just different mental models
- Clusters enable cooperation between blocks = faster algorithms! 🚀

**Essential API**:
```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
auto cluster = cg::this_cluster();
cluster.sync();  // Your new best friend! 😊
```

---

*Happy Cluster Programming! 🎉*
