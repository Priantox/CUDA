#include <stdio.h>

__global__ void hello_cuda(){
    printf("Hello cuda\n");
    printf("Block Index X: %d, Block Index Y: %d, Thread Index X: %d, Thread Index Y: %d\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main(){
    hello_cuda<<<2,2>>>(); //Block and thread
    cudaDeviceSynchronize();

    return 0;
}