/*
 ============================================================================
 Name        : parallel_CUDA_imp.cu
 Author      : Tomasz Pęcak and Paweł Renc
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>

//MODEL PARAMS

#define ROWS 100
#define COLS 100
#define CELL_SIZE_X 10
#define CELL_SIZE_Y 10
#define THICKNESS 50

#define Syinitial 0.1
#define Kinitial  0.0000125

#define headFixed 50
#define headCalculated 50

#define SIMULATION_ITERATIONS 1000
#define BLOCK_SIZE 256

double delta_t_ = 4000;
double qw = 0.001;

int posSy = ROWS / 2;
int posSx = COLS / 2;

struct CA {
    double *head;
    double *Sy;
    double *K;
    double *Source;
} h_ca, d_read, d_write;

double *d_write_head;

void init_host_ca();

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * CUDA kernel that computes simulation step
 */
__global__ void simulation_step_kernel(CA data, double *d_write_head) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ROWS * COLS) {
        d_write_head[idx] = data.head[idx] * 2;
    }
}

/**
 * Host function that copies the data
 */
void copy_data_from_CPU_to_GPU() {
    double *d_read_head, *d_read_Sy, *d_read_K, *d_read_Source;
    CUDA_CHECK_RETURN(cudaMalloc(&d_read_head, sizeof(double) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMalloc(&d_write_head, sizeof(double) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMalloc(&d_read_Sy, sizeof(double) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMalloc(&d_read_K, sizeof(double) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMalloc(&d_read_Source, sizeof(double) * ROWS * COLS));

    CUDA_CHECK_RETURN(cudaMemcpy(d_read_head, h_ca.head, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_write_head, h_ca.head, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_read_Sy, h_ca.Sy, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_read_K, h_ca.K, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_read_Source, h_ca.Source, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
}

void copy_data_from_GPU_to_CPU() {
    CUDA_CHECK_RETURN(cudaMemcpy(h_ca.head, d_write_head, sizeof(double) * ROWS * COLS, cudaMemcpyDeviceToHost));
}

void perform_simulation_on_GPU() {
    const int blockCount = (ROWS * COLS) / BLOCK_SIZE + 1;
    for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
        simulation_step_kernel << < blockCount, BLOCK_SIZE >> > (d_read, d_write_head);
    }
}

int main(void) {
    h_ca.head = new double[ROWS * COLS]();
    h_ca.Sy = new double[ROWS * COLS]();
    h_ca.K = new double[ROWS * COLS]();
    h_ca.Source = new double[ROWS * COLS]();

    init_host_ca();

    copy_data_from_CPU_to_GPU();

    printf("%lf", h_ca.head[100]);
    perform_simulation_on_GPU();

    copy_data_from_GPU_to_CPU();
    printf("%lf", h_ca.head[100]);
    return 0;
}

void init_host_ca() {
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++) {
            h_ca.head[i * ROWS + j] = headFixed;
            if (j == COLS - 1) {
                h_ca.head[i * ROWS + j] = headCalculated;
            }
            h_ca.Sy[i * ROWS + j] = Syinitial;
            h_ca.K[i * ROWS + j] = Kinitial;
            h_ca.Source[i * ROWS + j] = 0;
        }

    h_ca.Source[posSy * ROWS + posSx] = qw;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line
              << std::endl;
    exit(1);
}

