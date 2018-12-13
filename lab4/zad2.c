#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_blas.h> 
#define BILLION 1000000000.0

double **new_rand_matrix(int m, int n){

    double **A = (double **) calloc(m, sizeof(double *));
    for(int i=0; i<m; i++)
    {
        A[i] = (double *) calloc(n, sizeof(double));
    }

    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            A[i][j] = (double) rand()/RAND_MAX;
        }
    }

    return A; 
}

void destroy_matrix(int m, int n, double **A){
    
    for(int i=0; i<m; i++)
    {
        free(A[i]);
    }
    
    free(A);
}
    
        

double **new_zero_matrix(int m, int n){

    double **A = (double **) calloc(m, sizeof(double *));
    for(int i=0; i<m; i++)
    {
        A[i] = (double *) calloc(n, sizeof(double));
    }

    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            A[i][j] = 0;
        }
    }

    return A; 
}

void print_matrix(int m, int n, double **A){

    for(int i=0; i<m; i++){
        for(int j=0; j<m; j++){
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double** naive_mul_matrices(double **A, double **B, int m1, int n1, int m2, int n2){
    double **C = new_zero_matrix(m1, n2);

    for(int i=0; i<m1; i++){
        for(int j=0; j<n2; j++){
            for(int k=0; k<m2; k++){
                C[i][j] = A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

double** better_mul_matrices(double **A, double **B, int m1, int n1, int m2, int n2){
    double **C = new_zero_matrix(m1, n2);

    for(int i=0; i<m1; i++){
        for(int k=0; k<m2; k++){
            for(int j=0; j<n2; j++){
                C[i][j] = A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

double *rand_block(int size){

    double *block = calloc(size, sizeof(double));
    for(int i=0; i<size; i++){
        block[i] = (double) rand()/RAND_MAX;
    }
    return block;
}

double *zero_block(int size){

    double *block = calloc(size, sizeof(double));
    for(int i=0; i<size; i++){
        block[i] = 0;
    }
    return block;
}

int main(void){
    
    int SIZE = 5;
    srand(time(NULL));

    FILE *file = fopen("times_c.csv", "w+");
    if(file == NULL){
        perror("IO ERROR");
    }
    fprintf(file, "%s,%s,%s\n", "size", "version", "time");

    double naive_time;
    double better_time = 0.0;
    double blas_time = 0.0;

    struct timespec tp_start;
    struct timespec tp_stop;

    for(int i=100; i<=1000; i=i+100)
    {
        double **A = new_rand_matrix(i, i);
        double **B = new_rand_matrix(i, i);
        double **C;

        for(int j=0; j<10; j++){
            clock_gettime(CLOCK_REALTIME, &tp_start);
            C = naive_mul_matrices(A, B, i, i, i, i);
            clock_gettime(CLOCK_REALTIME, &tp_stop);

            naive_time = (double) tp_stop.tv_sec + tp_stop.tv_nsec/BILLION - tp_start.tv_sec - tp_start.tv_nsec/BILLION;

            clock_gettime(CLOCK_REALTIME, &tp_start);
            C = better_mul_matrices(A, B, i, i, i, i);
            clock_gettime(CLOCK_REALTIME, &tp_stop);

            better_time = (double) tp_stop.tv_sec + tp_stop.tv_nsec/BILLION - tp_start.tv_sec - tp_start.tv_nsec/BILLION;

            fprintf(file, "%d,%s,%f\n", i, "naive", naive_time);
            fprintf(file, "%d,%s,%f\n", i, "better", better_time);
        }

        destroy_matrix(i, i, A);
        destroy_matrix(i, i, B);
        destroy_matrix(i, i, C);

        double *A_block = rand_block(i*i);
        double *B_block = rand_block(i*i);
        double *C_block = zero_block(i*i);
        
        gsl_matrix_view A_blas = gsl_matrix_view_array(A_block, i, i);
        gsl_matrix_view B_blas = gsl_matrix_view_array(B_block, i, i);
        gsl_matrix_view C_blas = gsl_matrix_view_array(C_block, i, i);

        for(int j=0; j<10; j++){
            clock_gettime(CLOCK_REALTIME, &tp_start);
            gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                      1.0, &A_blas.matrix, &B_blas.matrix,
                      0.0, &C_blas.matrix);
            clock_gettime(CLOCK_REALTIME, &tp_stop);

            blas_time = (double) tp_stop.tv_sec + tp_stop.tv_nsec/BILLION - tp_start.tv_sec - tp_start.tv_nsec/BILLION;
            fprintf(file, "%d,%s,%f\n", i, "blas", blas_time);
        }


        
    }

    fclose(file);       
    return 0;
}



