# 🧠 CPU vs 🖥️ GPU

This document explains the key differences between a **CPU** and a **GPU** in a simple and easy-to-understand way.

---

## 🧠 CPU (Central Processing Unit)

The CPU is the **main brain of the computer**. It handles general tasks and controls the overall system.

### 🔑 Key Characteristics

- **🛠️ General Purpose**  
  Designed to perform many different types of tasks such as running applications, operating systems, and logic-heavy programs.

- **⚡ High Clock Speed**  
  Each core works very fast, allowing quick execution of individual tasks.

- **👥 Few Cores**  
  Usually has a small number of powerful cores (e.g., 4–16 cores).

- **🗂️ High Cache**  
  Has large and fast memory close to the cores to quickly access frequently used data.

- **⏱️ Low Latency**  
  Responds very quickly to tasks with minimal delay.

- **📉 Low Throughput**  
  Can execute tasks very fast, but cannot handle a huge number of tasks simultaneously.

### ✅ Summary
> The CPU is fast, flexible, and responsive, making it ideal for general-purpose computing and real-time tasks.

---

## 🖥️ GPU (Graphics Processing Unit)

The GPU is designed for **specialized, parallel processing**, especially for graphics and large-scale computations.

### 🔑 Key Characteristics

- **🎯 Specialized**  
  Optimized for specific tasks like graphics rendering, image processing, AI, and parallel computations.

- **🐢 Low Clock Speed**  
  Individual GPU cores are slower compared to CPU cores.

- **🏭 Many Cores**  
  Contains hundreds or thousands of smaller cores that work in parallel.

- **📦 Low Cache**  
  Has less cache per core since it focuses on processing large datasets rather than quick memory access.

- **⏳ High Latency**  
  Takes more time to start a task, making it less suitable for quick-response jobs.

- **📈 High Throughput**  
  Can process a massive amount of data simultaneously, making it ideal for parallel workloads.

### ✅ Summary
> The GPU is slower per task but extremely powerful when handling large amounts of similar work at the same time.

---

## ⚖️ Quick Comparison

| Feature        | 🧠 CPU | 🖥️ GPU |
|----------------|-------|--------|
| Purpose        | General-purpose | Specialized |
| Cores          | Few, powerful | Many, simple |
| Clock Speed    | High | Low |
| Latency        | Low | High |
| Throughput     | Low | High |
| Best For       | OS, apps, logic | Graphics, AI, CUDA |

---

## 🏁 Conclusion

- **🧠 CPU**: Best for tasks that need quick decisions and flexibility  
- **🖥️ GPU**: Best for tasks that can be done in parallel on large datasets



----
# 🧠 DRAM (Dynamic Random Access Memory)

**DRAM** stands for **Dynamic Random Access Memory**.  
It is the **main memory (RAM)** used by a computer to store data temporarily while programs are running.



## 📌 Simple Idea

DRAM works as the computer’s **short-term memory**.

- Stores data that the CPU and GPU are currently using
- Data is **temporary** and disappears when power is turned off

👉 Like writing notes on a whiteboard while solving a problem.

-----

## 🔄 Why It Is Called *Dynamic*

DRAM cells slowly lose their data, so they must be **refreshed continuously**.

- The system refreshes DRAM many times per second
- Without refreshing, stored data would disappear

👉 Like rewriting fading notes so they stay visible.

---

## ⚡ Why Computers Use DRAM

- Faster than storage devices (SSD/HDD)
- Cheaper and higher capacity than cache memory (SRAM)
- Can store large amounts of data (measured in GBs)

👉 A good balance between **speed, size, and cost**.

---

## 🔍 Key Characteristics

- **Volatile** → Data is lost when power is off  
- **Moderate Speed** → Slower than cache, faster than storage  
- **Large Capacity** → Stores gigabytes of data  
- **Main Memory** → Used directly by the CPU and GPU

---

## 🆚 DRAM vs Cache (Quick View)

| Feature | Cache (SRAM) | DRAM |
|-------|--------------|------|
| Speed | Very fast | Slower |
| Size | Very small | Large |
| Cost | Expensive | Cheaper |
| Use | Immediate data | Main memory |

---

## 🧠 Summary

> **DRAM is the main memory of a computer that temporarily stores data needed by running programs.**

---
---

# 🧠 CPU (Host) vs 🖥️ GPU (Device) — CUDA Programming Basics

This document explains how **CPU and GPU work together in CUDA**, what their goals are, and introduces important CUDA terms in a simple and detailed way.

---

## 🧠 CPU (Host)

The **CPU** is called the **host** in CUDA programming.  
It controls the program and handles decision-making and coordination.

### 🎯 Main Goal: Minimize Time of One Task
The CPU is designed to finish **one task as fast as possible**.

Examples:
- Running application logic
- Handling user input
- Managing memory and program flow

### 📏 Performance Metric: Latency (seconds)

**Latency** = time taken to complete a single task.

- Lower latency = faster response
- Very important for operating systems and interactive programs

👉 CPUs care about **how fast one task finishes**, not how many tasks run together.

---

## 🖥️ GPU (Device)

The **GPU** is called the **device** in CUDA programming.  
It is used for heavy computation and parallel processing.

### 🎯 Main Goal: Maximize Throughput
The GPU is designed to perform **many tasks at the same time**.

Examples:
- Processing millions of pixels
- Running thousands of mathematical calculations
- Training neural networks


### 📊 Performance Metric: Throughput (tasks per second)

**Throughput** = amount of work done per unit time.

Examples:
- Pixels per millisecond
- Calculations per second

👉 GPUs care about **total work done**, not speed of one single task.

---

## 🚀 Typical CUDA Program Workflow

A CUDA program usually follows these steps:

### 1️⃣ CPU Allocates Memory
The CPU allocates memory in **main system RAM** for input data.

---

### 2️⃣ CPU Copies Data to GPU
The CPU copies data from **RAM → GPU memory (VRAM)**.

👉 GPU cannot directly use CPU memory.

---

### 3️⃣ CPU Launches a Kernel
The CPU launches a **kernel**, which is a GPU function.

- CPU says *what* to run
- GPU decides *how many times* to run it in parallel

---

### 4️⃣ GPU Executes the Kernel
The GPU runs **many copies of the kernel simultaneously**.

👉 This is where parallel processing happens.

---

### 5️⃣ CPU Copies Results Back
After computation:
- GPU sends results back to CPU
- CPU uses results for display, storage, or further logic

---

## 🧩 What Is a Kernel?

A **kernel** is a function that:
- Runs on the GPU
- Is executed by **thousands of threads in parallel**

### Important Idea
A kernel looks like a **normal serial function**, but the GPU runs it many times at once.

👉 You do not manually write parallel loops for each thread — CUDA handles that.

---

## 🧩 Jigsaw Puzzle Analogy

Imagine solving a jigsaw puzzle:

- Each puzzle piece = one GPU thread
- Kernel instruction = “put this piece in the correct position”
- All pieces are placed **at the same time**
- Order does not matter, only the final picture

👉 This is exactly how GPU parallelism works.

---

## 🧠 Important CUDA Terms to Remember

### 🔹 Kernel
- A function that runs on the GPU
- Executed in parallel by many threads
- Different from Linux kernels or convolution kernels

---

### 🔹 Threads, Blocks, and Grid

- **Thread** → Does one small piece of work  
- **Block** → Group of threads  
- **Grid** → Group of blocks  

👉 Think of:
- Worker → Team → Factory

---

### 🔹 GEMM
**GEMM = General Matrix Multiplication**

- Core operation in GPUs
- Used in graphics, AI, and scientific computing

---

### 🔹 SGEMM
**SGEMM = Single-precision (FP32) GEMM**

- Uses 32-bit floating-point numbers
- Faster and commonly used in CUDA and deep learning

---

## 🧠 CPU vs GPU Naming in CUDA

| CPU | GPU |
|----|----|
| Host | Device |
| Executes functions | Executes kernels |
| Controls program | Performs computation |

👉 CPU = Manager  
👉 GPU = Massive parallel worker

---

## 🏁 Final Summary

- **CPU (Host)**  
  - Optimized for low latency  
  - Controls memory and program flow  

- **GPU (Device)**  
  - Optimized for high throughput  
  - Executes kernels in massive parallelism  

> **In CUDA, the CPU manages the work, and the GPU does the heavy computation.**
