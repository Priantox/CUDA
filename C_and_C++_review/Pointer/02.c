#include <stdio.h>

int main(){
    int num = 20;
    float fnum = 3.12;
    void* vptr = &num;

    printf("Integar: %d\n", *(int*)vptr);

    vptr = &fnum;
    printf("Float: 0.2f\n", *(float*)vptr);
}