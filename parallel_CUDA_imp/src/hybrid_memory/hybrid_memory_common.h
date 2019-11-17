#include "../params.h"
#include "../common/cuda_error_check.cu"
#include "../common/write_heads_to_file.c"
#include <iostream>
#include <numeric>
struct CA h_ca;

double *d_write_head;
CA *d_read_ca;

void copy_data_from_CPU_to_GPU() {
    double *d_read_head, *d_read_Sy, *d_read_K, *d_read_Source;

    CUDASAFECALL(cudaMalloc((void **) &d_read_ca, sizeof(*d_read_ca)));
    CUDASAFECALL(cudaMalloc((void **) &d_read_head, sizeof(*d_read_head) * ROWS * COLS));
    CUDASAFECALL(cudaMalloc((void **) &d_write_head, sizeof(double) * ROWS * COLS));
    CUDASAFECALL(cudaMalloc(&d_read_Sy, sizeof(double) * ROWS * COLS));
    CUDASAFECALL(cudaMalloc(&d_read_K, sizeof(double) * ROWS * COLS));
    CUDASAFECALL(cudaMalloc(&d_read_Source, sizeof(double) * ROWS * COLS));

    CUDASAFECALL(cudaMemcpy(d_read_head, h_ca.head, sizeof(*d_read_head) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDASAFECALL(cudaMemcpy(&(d_read_ca->head), &d_read_head, sizeof(d_read_ca->head), cudaMemcpyHostToDevice));

    CUDASAFECALL(cudaMemcpy(d_read_Sy, h_ca.Sy, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDASAFECALL(cudaMemcpy(&(d_read_ca->Sy), &d_read_Sy, sizeof(d_read_ca->Sy), cudaMemcpyHostToDevice));

    CUDASAFECALL(cudaMemcpy(d_read_K, h_ca.K, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDASAFECALL(cudaMemcpy(&(d_read_ca->K), &d_read_K, sizeof(d_read_ca->K), cudaMemcpyHostToDevice));

    CUDASAFECALL(cudaMemcpy(d_read_Source, h_ca.Source, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));
    CUDASAFECALL(
            cudaMemcpy(&(d_read_ca->Source), &d_read_Source, sizeof(d_read_ca->Source), cudaMemcpyHostToDevice));

    CUDASAFECALL(cudaMemcpy(d_write_head, h_ca.head, sizeof(double) * ROWS * COLS, cudaMemcpyHostToDevice));

}

void init_host_ca() {
    h_ca.head = new double[ROWS * COLS]();
    h_ca.Sy = new double[ROWS * COLS]();
    h_ca.K = new double[ROWS * COLS]();
    h_ca.Source = new double[ROWS * COLS]();

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

    int x,y;
    double source;
    for(int i = 0; i < NUMBER_OF_WELLS; i++){
    	x = wellsRows[i];
    	y = wellsCols[i];
    	source = wellsQW[i];
    	h_ca.Source[y *ROWS + x] = source;
    }
}

void copy_data_from_GPU_to_CPU() {
    CUDASAFECALL(cudaMemcpy(h_ca.head, d_write_head, sizeof(double) * ROWS * COLS, cudaMemcpyDeviceToHost));
}
