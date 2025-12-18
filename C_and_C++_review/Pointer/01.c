#include <stdio.h>

int main(){
    int value = 42;
    int *ptr1 = &value;
    int **ptr2 =  &ptr1;
    int ***ptr3 = &ptr2;

    int num = 20;
    float fnum = 3.12;
    void* vptr = &num;

    printf("Value: %d\n", ***ptr3);

    printf("Integar: %d\n", *(int*)vptr);

    vptr = &fnum;
    printf("Float: 0.2f\n", *(float*)vptr);
} 