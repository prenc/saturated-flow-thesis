#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

#include "params.h"

struct CA {
    double *head;
    double *Sy;
    double *K;
    double *Source;
} d_read, d_write;

void allocate_memory();
void init_read_ca();
void perform_simulation_on_GPU();
void write_heads_to_file();
void init_write_head();
void free_allocated_memory();


__device__ int active_cells_idx[ROWS*COLS];
__managed__ int dev_count = 0;

__device__ int my_push_back(int mt) {
  int insert_pt = atomicAdd(&dev_count, 1);
  if (insert_pt < ROWS*COLS){
	 active_cells_idx[insert_pt] = mt;
    return insert_pt;
  }
  else return -1;
}

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__global__ void simulation_step_kernel(struct CA d_ca, double *d_write_head) {
    unsigned ac_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned ac_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned ac_idx_g = ac_idx_y * COLS + ac_idx_x;

    if(ac_idx_g < dev_count){
        unsigned idx_g = active_cells_idx[ac_idx_g];
        unsigned idx_x = idx_g % COLS;
        unsigned idx_y = idx_g / COLS;
		if (idx_y != 0 && idx_y != ROWS - 1) {
				double Q = 0;
				double diff_head;
				double tmp_t;

				if (idx_x >= 1) {
					diff_head = d_ca.head[idx_g - 1] - d_ca.head[idx_g];
					tmp_t = d_ca.K[idx_g] * THICKNESS;
					Q += diff_head * tmp_t;
				}
				if (idx_y >= 1) {
					diff_head = d_ca.head[(idx_y - 1) * COLS + idx_x] - d_ca.head[idx_g];
					tmp_t = d_ca.K[idx_g] * THICKNESS;
					Q += diff_head * tmp_t;
				}
				if (idx_x + 1 < COLS) {
					diff_head = d_ca.head[idx_g + 1] - d_ca.head[idx_g];
					tmp_t = d_ca.K[idx_g] * THICKNESS;
					Q += diff_head * tmp_t;
				}
				if (idx_y + 1 < ROWS) {
					diff_head = d_ca.head[(idx_y + 1) * COLS + idx_x] - d_ca.head[idx_g];
					tmp_t = d_ca.K[idx_g] * THICKNESS;
					Q += diff_head * tmp_t;
				}

				Q -= d_ca.Source[idx_g];

				double ht1 = Q * DELTA_T;
				double ht2 = AREA * d_ca.Sy[idx_g];

			  d_write_head[idx_g] = d_ca.head[idx_g] + ht1 / ht2;
		}
    }
}
__global__ void find_active_cells_kernel(struct CA d_ca) {
    unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx_g = idx_y * COLS + idx_x;

    if(idx_x < ROWS && idx_y < COLS ){
		if(d_ca.head[idx_g]  < headFixed || d_ca.Source[idx_g] != 0 ){
			my_push_back(idx_g);
			return;
		}
		if (idx_x >= 1) {
			if(d_ca.head[idx_g - 1] < headFixed){
				my_push_back(idx_g);
				return;
			}
		}
		if (idx_y >= 1) {
			if(d_ca.head[idx_g - COLS] < headFixed){
				my_push_back(idx_g);
				return;
			}
		}
		if (idx_x + 1 < COLS) {
			if(d_ca.head[idx_g + 1] < headFixed){
				my_push_back(idx_g);
				return;
			}
		}
		if (idx_y + 1 < ROWS) {
			if(d_ca.head[idx_g + COLS] < headFixed){
				my_push_back(idx_g);
				return;
			}
		}
    }
}

int main(void) {
    allocate_memory();

    init_read_ca();

    init_write_head();

    perform_simulation_on_GPU();

    write_heads_to_file();

    free_allocated_memory();

    return 0;
}

void allocate_memory() {
    CUDA_CHECK_RETURN(cudaMallocManaged(&(d_read.head), sizeof(double) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMallocManaged(&(d_write.head), sizeof(double) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMallocManaged(&(d_read.Sy), sizeof(double) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMallocManaged(&(d_read.K), sizeof(double) * ROWS * COLS));
    CUDA_CHECK_RETURN(cudaMallocManaged(&(d_read.Source), sizeof(double) * ROWS * COLS));
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


void perform_simulation_on_GPU() {
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	const int blockCount = (ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE) + 1;
	double gridSize = sqrt(blockCount) + 1;
	dim3 blockCount2D(gridSize, gridSize);
	double activeBlockCount, activeGridSize;
	for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
		if(dev_count < ROWS*COLS ){
			dev_count = 0;
			find_active_cells_kernel << < blockCount2D, blockSize >> > (d_read);
			cudaDeviceSynchronize();
		}
		activeBlockCount = dev_count* dev_count/ (BLOCK_SIZE * BLOCK_SIZE);
		activeGridSize = sqrt(activeBlockCount) + 1;
		dim3 activeBlockCount2D(activeGridSize, activeGridSize);
		simulation_step_kernel << < activeBlockCount2D, blockSize >> > (d_read, d_write.head);
		cudaDeviceSynchronize();

		double *tmp1 = d_write.head;
		d_write.head = d_read.head;
		d_read.head = tmp1;
    }
}

void write_heads_to_file() {
    FILE *fp;
    fp = fopen("heads_ca.txt", "w");

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            fprintf(fp, "%lf, ", d_write.head[i * ROWS + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void free_allocated_memory(){
	cudaFree(d_read.head);
	cudaFree(d_write.head);
	cudaFree(d_read.Sy);
	cudaFree(d_read.K);
	cudaFree(d_read.Source);
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
