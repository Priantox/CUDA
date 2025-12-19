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

    // Safe to use ptr after NULL check
    *ptr = 42;
    printf("5. Value at ptr: %d\n", *ptr);

    //Free allocated memory
    free(ptr);
    ptr = NULL;

    if (ptr == NULL) {
    printf("7. ptr is NULL, safely avoided use after free\n");
    }


    return 0;
}