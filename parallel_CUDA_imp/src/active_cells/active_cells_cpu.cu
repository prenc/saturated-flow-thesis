#include "active_cells_common.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

thrust::device_vector<int> d_active_cells_vector;

__global__ void simulation_step_kernel(struct CA d_ca, double *d_write_head, int *ac_array, int ac_array_size) {
    unsigned ac_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned ac_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned ac_idx_g = ac_idx_y * COLS + ac_idx_x;

    if(ac_idx_g < ac_array_size){
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

void find_active_cells(){
	thrust::host_vector<int> h_active_cells_vector;
	for(int i = 0; i < ROWS; i++){
		for(int j = 0; j < ROWS; j++){
			int idx_g = j* COLS + i;
			if(d_read.head[idx_g] < headFixed){
				h_active_cells_vector.push_back(idx_g);
				if(i + 1 < ROWS) h_active_cells_vector.push_back(idx_g + 1);
				if(i - 1 >= 0 )   h_active_cells_vector.push_back(idx_g - 1);
				if(j + 1 < COLS) h_active_cells_vector.push_back(idx_g + COLS);
				if(j - 1 >= 0)    h_active_cells_vector.push_back(idx_g - COLS);
			}else if(d_read.Source[idx_g] != 0){
				h_active_cells_vector.push_back(idx_g);
			}
		}
	}
	thrust::host_vector<int> h_active_cells_vector_result;
	thrust::sort(h_active_cells_vector.begin(),h_active_cells_vector.end());
	int previous_value = h_active_cells_vector[0];
	h_active_cells_vector_result.push_back(previous_value);
	for(int i = 1; i< h_active_cells_vector.size(); i++){
		if(h_active_cells_vector[i] != previous_value){
			previous_value = h_active_cells_vector[i];
			h_active_cells_vector_result.push_back(previous_value);
		}
	}
	d_active_cells_vector = h_active_cells_vector_result;
}

void perform_simulation_on_GPU() {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
    	find_active_cells();
    	int* active_cells_array = thrust::raw_pointer_cast( &d_active_cells_vector[0] );
    	int size = d_active_cells_vector.size();
        const int blockCount = size * size / (BLOCK_SIZE * BLOCK_SIZE);
        double gridSize = sqrt(blockCount) + 1;
        dim3 blockCount2D(gridSize, gridSize);
        simulation_step_kernel << < blockCount2D, blockSize >> > (d_read, d_write.head, active_cells_array, size );

        cudaDeviceSynchronize();

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

    free_allocated_memory();
    return 0;
}

