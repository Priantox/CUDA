#include <stdio.h>
#include <stdlib.h>

int main(){
    int arr[] = {10, 20, 30, 40, 50};

    //size_t
    size_t size = sizeof(arr) / sizeof(arr[0]);
    printf("Size of arr: %zu\n", size);  // %zu -> size_t
    printf("size of size_t: %zu\n", sizeof(size_t));
    printf("int size in bytes: %zu\n", sizeof(int));

    typedef struct
    {
        float x;
        float y;
    } Point;
    
    Point p = {1.1, 2.5};
    printf("Size of Point: %zu\n", sizeof(Point));

    return 0;
}