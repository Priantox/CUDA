# 🚀 CUDA Thread & Block Basics (Simple Explanation)

This README explains common CUDA terms in a **short and easy way**.

---

## 📦 `blockIdx`
- Tells **which block** is currently running.
- It is the block’s position inside the grid.

---

## 📐 `blockDim`
- Tells **how many threads are inside one block**.
- Same for every block.

---

## 🆔 `block_id`
- A **single unique number** for a block.
- Often calculated from `blockIdx`.

---

## 🧵 `thread_offset`
- The position of a thread **inside its own block**.
- Usually `threadIdx.x`.

---

## ➕ `block_offset`
- Number of threads **before this block starts**.
- Calculated as:

---