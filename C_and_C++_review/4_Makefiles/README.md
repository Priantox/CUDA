## **Compiling and Running Programs**

### **Using Makefile**

Open terminal (WSL), navigate to the project folder:

```bash
cd /path/to/C_and_C++_review/4_Makefiles
```

# Compile C program 01

```
make 01
./01     
```

# Compile C program 02

```
make 02
./02
```

# Compile CUDA program 03

```
make 03
./03_cu
```

# C programs

```
gcc -o 01 01.c
./01
gcc -o 02 02.c
./02
```

# CUDA program

```
nvcc -arch=sm_86 -o 03_cu 03.cu
./03_cu
```

# Make clean or manually:

```
rm -f 01 02 03_cu *.o
```
