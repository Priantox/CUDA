#include<stdio.h>
#include<stdlib.h>

int main(){
    //Initialize pointer to null
    int* ptr = NULL;
    printf("Initialize ptr value: %p\n", (void*)ptr);

    if(ptr == NULL){
        printf("Ptr is NULL, cannot dereference\n");
    }

    //Alocate memory
    ptr = malloc(sizeof(int));
    if(ptr == NULL){
        printf("Memory allocation failed\n");
        return 1;
    }

    return 0;
}