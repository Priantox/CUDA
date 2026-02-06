#include <stdio.h>
#include <stdlib.h>

void matrix_vector_product(float *matrix, float *v1, float *v2, int matrix_size){
    for(int i = 0; i < matrix_size; i++){
        float sum = 0.0f;
        for(int j = 0; j < matrix_size; j++){
            sum += matrix[i * matrix_size + j] * v1[j];
        }
        v2[i] = sum;
    }
}

int main(int argc, char **argv){
    float *matrix, *v1, *v2;

    int matrix_size = 4000;

    matrix = (float *) malloc(matrix_size * matrix_size * sizeof(float));
    v1 = (float *) malloc(matrix_size * sizeof(float));
    v2 = (float *) malloc(matrix_size * sizeof(float));

    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            matrix[i * matrix_size + j] = (float) i*matrix_size + j;
        }
    }

    for(int i = 0; i < matrix_size; i++)
        v1[i] = (float) i;
    
    matrix_vector_product(matrix, v1, v2, matrix_size);

    for(int i = 0; i < matrix_size; i++)
        printf("%0.2f\n", v2[i]);
    
    free(matrix);
    free(v1);
    free(v2);

    return 0;
}

