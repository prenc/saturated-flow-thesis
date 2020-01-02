#include "active_cells_common.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

thrust::device_vector<int> d_active_cells_vector;

__global__ void simulation_step_kernel(struct CA d_ca, double *d_write_head, int *ac_array, int ac_array_size, int ac_grid_size) {
	unsigned ac_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned ac_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned ac_idx_g = ac_idx_y * blockDim.x * ac_grid_size + ac_idx_x;

	if (ac_idx_g < ac_array_size) {
		unsigned idx_g = ac_array[ac_idx_g];
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
bool check_if_cell_is_active(int idx_x, int idx_y){
	int idx_g = idx_y * COLS + idx_x;
	if(d_read.head[idx_g]  < headFixed || d_read.Source[idx_g] != 0 ){
		return true;
	}
	if (idx_x >= 1) {
		if(d_read.head[idx_g - 1] < headFixed){
			return true;
		}
	}
	if (idx_y >= 1) {
		if(d_read.head[idx_g - COLS] < headFixed){
			return true;
		}
	}
	if (idx_x + 1 < COLS) {
		if(d_read.head[idx_g + 1] < headFixed){
			return true;
		}
	}
	if (idx_y + 1 < ROWS) {
		if(d_read.head[idx_g + COLS] < headFixed){
			return true;
		}
	}
	return false;
}

void find_active_cells() {
	thrust::host_vector<int> h_active_cells_vector;
	for (int idx_y = 0; idx_y < ROWS; idx_y++) {
		for (int idx_x = 0; idx_x < ROWS; idx_x++) {
			int idx_g = idx_y * COLS + idx_x;
			if (check_if_cell_is_active(idx_x, idx_y)) {
				h_active_cells_vector.push_back(idx_g);
			}
		}
	}
	d_active_cells_vector = h_active_cells_vector;
}

void perform_simulation_on_GPU() {
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

	int blockCount = ceil((double) ROWS * COLS / (BLOCK_SIZE * BLOCK_SIZE));
	int gridSize = ceil(sqrt(blockCount));
	dim3 gridDim(gridSize, gridSize);

	int *active_cells_array;
	int active_cell_vector_size = 0, activeBlockCount, activeGridSize;
	for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
		if(active_cell_vector_size < ROWS * COLS){
			printf("Iteration number: %d\n",i);
			find_active_cells();
			active_cells_array = thrust::raw_pointer_cast(&d_active_cells_vector[0]);
			active_cell_vector_size = d_active_cells_vector.size();

			activeBlockCount = ceil(double(active_cell_vector_size) / (BLOCK_SIZE * BLOCK_SIZE));
			activeGridSize = ceil(sqrt(activeBlockCount));
			dim3 activeGridDim(activeGridSize, activeGridSize);

			simulation_step_kernel << < activeGridDim, blockSize >> > (d_read, d_write.head, active_cells_array, active_cell_vector_size, activeGridSize);
			cudaDeviceSynchronize();
		}else{
			simulation_step_kernel << < gridDim, blockSize >> > (d_read, d_write.head, active_cells_array, active_cell_vector_size, gridSize);
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

