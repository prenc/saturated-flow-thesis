#include "global_memory_common.h"

__global__ void simulation_step_kernel(struct CA *d_ca, double *d_write_head) {
	unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned idx_g = idx_y * COLS + idx_x;

	double Q, diff_head, tmp_t, ht1, ht2;
	if (idx_x < COLS && idx_y < ROWS) {
		if (idx_y != 0 && idx_y != ROWS - 1) {
			Q = 0;
			if (idx_x >= 1) {
				diff_head = d_ca->head[idx_g - 1] - d_ca->head[idx_g];
				tmp_t = d_ca->K[idx_g] * THICKNESS;
				Q += diff_head * tmp_t;
			}
			if (idx_y >= 1) {
				diff_head = d_ca->head[(idx_y - 1) * COLS + idx_x] - d_ca->head[idx_g];
				tmp_t = d_ca->K[idx_g] * THICKNESS;
				Q += diff_head * tmp_t;
			}
			if (idx_x + 1 < COLS) {
				diff_head = d_ca->head[idx_g + 1] - d_ca->head[idx_g];
				tmp_t = d_ca->K[idx_g] * THICKNESS;
				Q += diff_head * tmp_t;
			}
			if (idx_y + 1 < ROWS) {
				diff_head = d_ca->head[(idx_y + 1) * COLS + idx_x] - d_ca->head[idx_g];
				tmp_t = d_ca->K[idx_g] * THICKNESS;
				Q += diff_head * tmp_t;
			}

			Q -= d_ca->Source[idx_g];
			ht1 = Q * DELTA_T;
			ht2 = AREA * d_ca->Sy[idx_g];

			d_write_head[idx_g] = d_ca->head[idx_g] + ht1 / ht2;
			if (d_write_head[idx_g] < 0) {
				d_write_head[idx_g] = 0;
			}
		}
	}
}

void perform_simulation_on_GPU() {
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	const int blockCount = ceil((ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
	int gridSize = ceil(sqrt(blockCount));
	dim3 gridDim(gridSize, gridSize);
	for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
		simulation_step_kernel << < gridDim, blockSize >> > (d_read_ca, d_write_head);

		cudaDeviceSynchronize();

		double *tmp1 = d_write_head;
		CUDASAFECALL(
				cudaMemcpy(&d_write_head, &(d_read_ca->head), sizeof(d_read_ca->head), cudaMemcpyDeviceToHost));
		CUDASAFECALL(cudaMemcpy(&(d_read_ca->head), &tmp1, sizeof(tmp1), cudaMemcpyHostToDevice));
	}
}

int main(int argc, char *argv[]) {
	init_host_ca();
	copy_data_from_CPU_to_GPU();

	perform_simulation_on_GPU();

	if(WRITE_OUTPUT_TO_FILE){
		copy_data_from_GPU_to_CPU();
		write_heads_to_file(h_ca.head, argv[0]);
	}

    return 0;
}
