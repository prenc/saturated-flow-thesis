#include "../../params.h"
#include "../common/memory_management.cuh"
#include "../common/statistics.h"

__global__ void simulation_step_kernel(struct CA *d_ca, double *d_write_head) {
    unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned idx_g = idx_y * blockDim.y * gridDim.x + idx_x;

    double Q, diff_head, tmp_t, ht1, ht2;

	for (int i = 0; i < KERNEL_LOOP_SIZE; i++) {
		if (idx_g < ROWS * COLS) {
			unsigned idx_x = idx_g % COLS;
			unsigned idx_y = idx_g / COLS;

			Q = 0;
			if (idx_y != 0 && idx_y != ROWS - 1) {
				if (idx_x >= 1) {
					diff_head = d_ca->heads[idx_g - 1] - d_ca->heads[idx_g];
					tmp_t = d_ca->K[idx_g] * THICKNESS;
					Q += diff_head * tmp_t;
				}
				if (idx_y >= 1) {
					diff_head = d_ca->heads[(idx_y - 1) * COLS + idx_x] - d_ca->heads[idx_g];
					tmp_t = d_ca->K[idx_g] * THICKNESS;
					Q += diff_head * tmp_t;
				}
				if (idx_x + 1 < COLS) {
					diff_head = d_ca->heads[idx_g + 1] - d_ca->heads[idx_g];
					tmp_t = d_ca->K[idx_g] * THICKNESS;
					Q += diff_head * tmp_t;
				}
				if (idx_y + 1 < ROWS) {
					diff_head = d_ca->heads[(idx_y + 1) * COLS + idx_x] - d_ca->heads[idx_g];
					tmp_t = d_ca->K[idx_g] * THICKNESS;
					Q += diff_head * tmp_t;
				}
			}

			Q -= d_ca->sources[idx_g];
			ht1 = Q * DELTA_T;
			ht2 = AREA * d_ca->Sy[idx_g];

			d_write_head[idx_g] = d_ca->heads[idx_g] + ht1 / ht2;
			if (d_write_head[idx_g] < 0) {
				d_write_head[idx_g] = 0;
			}
		}
    }
}

void perform_simulation_on_GPU() {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    double gridSize = ceil(sqrt(blockCount));
    dim3 gridDim(gridSize, gridSize);

	Timer stepTimer;
	startTimer(&stepTimer);

	for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
        simulation_step_kernel <<< gridDim, blockSize >>> (d_read_ca, d_write_head);

        cudaDeviceSynchronize();

        double *tmp1 = d_write_head;
        ERROR_CHECK(
                cudaMemcpy(&d_write_head, &(d_read_ca->heads), sizeof(d_read_ca->heads), cudaMemcpyDeviceToHost));
        ERROR_CHECK(cudaMemcpy(&(d_read_ca->heads), &tmp1, sizeof(tmp1), cudaMemcpyHostToDevice));

        if (i % STATISTICS_WRITE_FREQ == 0) {
		    endTimer(&stepTimer);
		    stats[i].stepTime = getElapsedTime(stepTimer);
		    startTimer(&stepTimer);
	    }
    }
}

int main(int argc, char *argv[]) {
    initializeCA();

    copyDataFromCpuToGpu();

    perform_simulation_on_GPU();

    copyDataFromGpuToCpu();

	if(WRITE_OUTPUT_TO_FILE){
        copyDataFromGpuToCpu();
		write_heads_to_file(h_ca.heads, argv[0]);
	}

	if (WRITE_STATISTICS_TO_FILE) {
		write_statistics_to_file( argv[0]);
	}

    return 0;
}



