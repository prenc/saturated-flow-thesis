#include "active_cells_common.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

__device__ int active_cells_idx[ROWS*COLS];
__managed__ int dev_active_cells_count = 0;

__device__ int my_push_back(int mt) {
  int insert_pt = atomicAdd(&dev_active_cells_count, 1);
  if (insert_pt < ROWS*COLS){
	 active_cells_idx[insert_pt] = mt;
    return insert_pt;
  }
  else return -1;
}

__global__ void simulation_step_kernel(struct CA d_ca, double *d_write_head) {
	int activeBlockCount = ceil((double)dev_active_cells_count / (BLOCK_SIZE * BLOCK_SIZE)) ;
	int activeGridSize = ceil(sqrtf(activeBlockCount));

	unsigned ac_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned ac_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned ac_idx_g = ac_idx_y * blockDim.x * activeGridSize + ac_idx_x;

    if(ac_idx_g < dev_active_cells_count){
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

void perform_simulation_on_GPU() {
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	const int blockCount = ceil((ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
	int gridSize = ceil(sqrt(blockCount)) ;
	dim3 gridDim(gridSize, gridSize);
	int activeBlockCount, activeGridSize;
	for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
		if(dev_active_cells_count < ROWS*COLS ){
			dev_active_cells_count = 0;
			find_active_cells_kernel << < gridDim, blockSize >> > (d_read);
			cudaDeviceSynchronize();

			activeBlockCount = ceil((double)dev_active_cells_count / (BLOCK_SIZE * BLOCK_SIZE)) ;
			activeGridSize = ceil(sqrt(activeBlockCount));
			dim3 activeGridDim(activeGridSize, activeGridSize);

			simulation_step_kernel << < activeGridDim, blockSize >> > (d_read, d_write.head);
			cudaDeviceSynchronize();
		}else{
			simulation_step_kernel << < gridDim, blockSize >> > (d_read, d_write.head);
			cudaDeviceSynchronize();
		}
		double *tmp1 = d_write.head;
		d_write.head = d_read.head;
		d_read.head = tmp1;
    }
}

int main(void) {
    allocate_memory();
    init_read_ca();
    init_write_head();

    perform_simulation_on_GPU();

	return 0;
}








