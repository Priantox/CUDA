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

