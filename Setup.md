# CUDA 13.1 Installation on WSL2 (Ubuntu)

This guide explains how to install **CUDA 13.1 Toolkit** on **WSL2** with an NVIDIA GPU.
⚠️ **Do NOT install NVIDIA Linux drivers** in WSL2 — WSL uses the Windows GPU driver.

---

## Prerequisites

- WSL2 installed and running (Ubuntu recommended)
- NVIDIA GPU with latest Windows driver installed
- Internet connection

## Step 1: Download CUDA Installer

## Check WSL version:

```bash
wsl --version


# Move to Linux home directory (safe)
cd ~

# Download CUDA 13.1 installer
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run


chmod +x cuda_13.1.0_590.44.01_linux.run
```

## Step 2: Make Installer Executable

```
chmod +x cuda_13.1.0_590.44.01_linux.run
```

## Step 3: Run CUDA Installe

```
sudo sh cuda_13.1.0_590.44.01_linux.run
```

## Step 4: Configure Environment Variables

--> Append CUDA paths to .bashrc

```
echo 'export PATH=/usr/local/cuda-13.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

```

--> Apply changes

```
source ~/.bashrc
```

## Step 5: Verify Installation

Check CUDA compiler:

```
nvcc --version

```

Check GPU access:

```
nvidia-smi

```
