<!-- Premium Badge Collection -->
![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.1-FF6B00?style=for-the-badge&logo=nvidia&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-WSL2-0078D4?style=for-the-badge&logo=windows&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-x86__64-FF0000?style=for-the-badge&logo=amd&logoColor=white)

<h1>⚡ Compute Unified Device Architecture ⚡</h1>

<h3>🎯 Professional CUDA 13.1 Development Environment Setup</h3>

<p align="center">
<strong>A comprehensive installation guide for CUDA 13.1 Toolkit on WSL2 with NVIDIA GPU acceleration</strong>
</p>

---

</div>

## 📋 Overview

> 🚀 This guide provides **step-by-step instructions** to install **CUDA 13.1 Toolkit** on **Windows Subsystem for Linux 2 (WSL2)** with full NVIDIA GPU support.

### ⚠️ Important Notice
> **Do NOT install NVIDIA Linux drivers in WSL2** — WSL2 uses the Windows GPU driver automatically through the GPU paravirtualization layer.

---

## 🔧 System Requirements

<table align="center" width="80%">
<tr>
<th width="30%">Component</th>
<th width="70%">Requirement</th>
</tr>
<tr>
<td><strong>🖥️ Operating System</strong></td>
<td>Windows 10 (Build 19041+) or Windows 11 with WSL2 enabled</td>
</tr>
<tr>
<td><strong>🐧 Linux Distribution</strong></td>
<td>Ubuntu 22.04 LTS or 20.04 LTS (recommended)</td>
</tr>
<tr>
<td><strong>🎮 GPU</strong></td>
<td>NVIDIA GPU with CUDA Compute Capability 3.5+</td>
</tr>
<tr>
<td><strong>🔌 Driver</strong></td>
<td>Latest NVIDIA Windows driver (495.29+)</td>
</tr>
</table>

---

# 🚀 Installation Process

## 📥 Step 1: Download CUDA Installer

### Verify WSL2 version
```bash
wsl --version
```
### Check current Linux distribution
```
cat /etc/os-release
```
### Verify GPU visibility
```
nvidia-smi
```

</details>

<details>
<summary><b>📦 Download Process</b></summary>

```bash
# Navigate to safe directory
cd ~

# Download CUDA 13.1 installer (1.6GB)
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run

# Verify download integrity (optional)
ls -la cuda_13.1.0_590.44.01_linux.run
```

</details>

---

### 🔐 Step 2: Prepare Installation

```bash
# Make installer executable
chmod +x cuda_13.1.0_590.44.01_linux.run

# Verify permissions
ls -la cuda_13.1.0_590.44.01_linux.run
```

---

### ⚙️ Step 3: Execute CUDA Installation

```bash
# Run CUDA installer with administrative privileges
sudo sh cuda_13.1.0_590.44.01_linux.run
```

<div align="center">

> 💡 **Installation Tips:**
> - Accept the End User License Agreement
> - **Uncheck** the Driver installation option
> - Select CUDA Toolkit, samples, and documentation
> - Use default installation path: `/usr/local/cuda-13.1`

</div>

---

### 🛠️ Step 4: Configure System Environment

<details>
<summary><b>🔧 Environment Variables Setup</b></summary>

```bash
# Add CUDA binaries to PATH
echo 'export PATH=/usr/local/cuda-13.1/bin:$PATH' >> ~/.bashrc

# Add CUDA libraries to library path
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Add CUDA_HOME for compatibility
echo 'export CUDA_HOME=/usr/local/cuda-13.1' >> ~/.bashrc

# Reload environment variables
source ~/.bashrc
```

</details>

---

### ✅ Step 5: Installation Verification

<table align="center" width="90%">
<tr>
<th width="50%">🔍 Verification Command</th>
<th width="50%">📊 Expected Output</th>
</tr>
<tr>
<td>

```bash
nvcc --version
```

</td>
<td>

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Mon_Apr__3_23:27:21_PDT_2024
Cuda compilation tools, release 13.1, V13.1.xxx
```

</td>
</tr>
<tr>
<td>

```bash
nvidia-smi
```

</td>
<td>

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 13.1   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA ...         Off  | 00000000:xx:xx.x Off |                  N/A |
+-------------------------------+----------------------+----------------------+
```

</td>
</tr>
</table>

---



## 🎉 Installation Complete!



---


