#include "unified_memory_common.h"

struct Statistics stats[ROWS*COLS];

__global__ void simulation_step_kernel(struct CA d_ca, double *d_write_head) {
    unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx_g = idx_y * COLS + idx_x;

    double Q, diff_head, tmp_t, ht1, ht2;

    if (idx_x < COLS && idx_y < ROWS) {
        if (idx_y != 0 && idx_y != ROWS - 1) {
            for (int i = 0; i < KERNEL_LOOP_SIZE; i++) {
	            if (i == SIMULATION_ITERATIONS - 1){
		            Q = 0;
	            }
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
            }

            Q -= d_ca.Source[idx_g];
            ht1 = Q * DELTA_T;
            ht2 = AREA * d_ca.Sy[idx_g];

            d_write_head[idx_g] = d_ca.head[idx_g] + ht1 / ht2;
        }
    }
}

void perform_simulation_on_GPU() {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    double gridSize = ceil(sqrt(blockCount));
    dim3 blockCount2D(gridSize, gridSize);

	Timer stepTimer;

	for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
	    startTimer(&stepTimer);

	    simulation_step_kernel << < blockCount2D, blockSize >> > (d_read, d_write.head);
        cudaDeviceSynchronize();

        double *tmp1 = d_write.head;
        d_write.head = d_read.head;
        d_read.head = tmp1;

	    endTimer(&stepTimer);
	    stats[i].stepTime = getElapsedTime(stepTimer);
    }
}

int main(int argc, char *argv[]) {
    allocate_memory();
    init_read_ca();
    init_write_head();

    perform_simulation_on_GPU();

	if(WRITE_OUTPUT_TO_FILE){
		write_heads_to_file(d_write.head, argv[0]);
	}

	if (WRITE_STATISTICS_TO_FILE) {
		write_statistics_to_file(stats, argv[0]);
	}

    return 0;
}




