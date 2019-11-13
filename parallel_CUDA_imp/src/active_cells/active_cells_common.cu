#include "../params.h"
#include "../common/cuda_error_check.cu"
#include "../common/write_heads_to_file.c"

struct CA d_read, d_write;

void allocate_memory() {
	CUDASAFECALL(cudaMallocManaged(&(d_read.head), sizeof(double) * ROWS * COLS));
	CUDASAFECALL(cudaMallocManaged(&(d_write.head), sizeof(double) * ROWS * COLS));
	CUDASAFECALL(cudaMallocManaged(&(d_read.Sy), sizeof(double) * ROWS * COLS));
	CUDASAFECALL(cudaMallocManaged(&(d_read.K), sizeof(double) * ROWS * COLS));
	CUDASAFECALL(cudaMallocManaged(&(d_read.Source), sizeof(double) * ROWS * COLS));
}

void init_read_ca() {
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++) {
            d_read.head[i * ROWS + j] = headFixed;
            if (j == COLS - 1) {
                d_read.head[i * ROWS + j] = headCalculated;
            }
            d_read.Sy[i * ROWS + j] = Syinitial;
            d_read.K[i * ROWS + j] = Kinitial;
            d_read.Source[i * ROWS + j] = 0;
        }

    d_read.Source[posSy * ROWS + posSx] = qw;
}

void init_write_head(){
	memcpy(d_write.head, d_read.head, sizeof(double)*ROWS*COLS);
}

void free_allocated_memory(){
	cudaFree(d_read.head);
	cudaFree(d_write.head);
	cudaFree(d_read.Sy);
	cudaFree(d_read.K);
	cudaFree(d_read.Source);
}
