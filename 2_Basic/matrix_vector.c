#include <stdio.h>
#include <stdlib.h>

void matrix_vector_product(float *matrix, float *v1, float *v2, int matrix_size){
    for(int i = 0; i < matrix_size; i++){
        float sum = 0.0f;
        for(int j = 0; j < matrix_size; j++){
            sum += A[i * matrix_size + j] * v1[j];
        }
        v2[i] = sum;
    }
}

int main(int argc, char **argv){
    float *A, *v1, *v2;

    int matrix_size = 3;

    A = (float *) malloc(matrix_size * matrix_size * sizeof(float));
    v1 = (float *) malloc(matrix_size * sizeof(float));
    v2 = (float *) malloc(matrix_size * sizeof(float));

    
}

