#include "../../params.h"
#include "../common/cuda_error_check.cu"
#include "../common/write_heads_to_file.c"
#include <iostream>
#include <numeric>
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

    int x,y;
    double source;
    for(int i = 0; i < NUMBER_OF_WELLS; i++){
    	x = wellsRows[i];
    	y = wellsCols[i];
    	source = wellsQW[i];
    	d_read.Source[y *ROWS + x] = source;
    }
}

void init_write_head() {
	memcpy(d_write.head, d_read.head, sizeof(double)*ROWS*COLS);
}

void free_allocated_memory(){
	cudaFree(d_read.head);
	cudaFree(d_write.head);
	cudaFree(d_read.Sy);
	cudaFree(d_read.K);
	cudaFree(d_read.Source);
}

void write_coverage_to_file(double *coverage_vector) {
	std::string output_path= "./output/";
	create_output_dir(output_path);
	std::string fileName = output_path + "coverage" + 
                           "_" +
	                       std::to_string(CA_SIZE) +
                           "_" +
                           std::to_string(SIMULATION_ITERATIONS) +
                           "_" +
                           std::to_string(NUMBER_OF_WELLS) +
                           ".csv";

	FILE *fp;
	fp = fopen(fileName.c_str(), "w");
	fprintf(fp, "Step, Coverage\n");
	for (int i = COVERAGE_WRITE_FREQ - 1; i <= SIMULATION_ITERATIONS; i+=COVERAGE_WRITE_FREQ) {
		fprintf(fp, "%d, %.2lf \n", i + 1, coverage_vector[i]);
	}
	fclose(fp);
}
