#include "../common/unified_memory_management.cuh"
#include "../common/statistics.h"

__global__ void simulation_step_kernel(struct CA d_ca, double *d_write_head) {
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx_g = y * blockDim.y * gridDim.x + x;

    double Q, diff_head, tmp_t, ht1, ht2;
    if (idx_g < ROWS * COLS) {
        unsigned idx_x = idx_g % COLS;
        unsigned idx_y = idx_g / COLS;

        if (idx_y != 0 && idx_y != ROWS - 1) {
            Q = 0;
            if (idx_x >= 1) {
                diff_head = d_ca.heads[idx_g - 1] - d_ca.heads[idx_g];
                tmp_t = d_ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_y >= 1) {
                diff_head = d_ca.heads[(idx_y - 1) * COLS + idx_x] - d_ca.heads[idx_g];
                tmp_t = d_ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_x + 1 < COLS) {
                diff_head = d_ca.heads[idx_g + 1] - d_ca.heads[idx_g];
                tmp_t = d_ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_y + 1 < ROWS) {
                diff_head = d_ca.heads[(idx_y + 1) * COLS + idx_x] - d_ca.heads[idx_g];
                tmp_t = d_ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }

            Q -= d_ca.sources[idx_g];
            ht1 = Q * DELTA_T;
            ht2 = AREA * d_ca.Sy[idx_g];

            d_write_head[idx_g] = d_ca.heads[idx_g] + ht1 / ht2;
	        if (d_write_head[idx_g] < 0) {
		        d_write_head[idx_g] = 0;
	        }
        }
    }
}

void perform_simulation_on_GPU() {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    double gridSize = ceil(sqrt(blockCount));
    dim3 gridDim(gridSize, gridSize);

    Timer stepTimer;
    startTimer(&stepTimer);

    for (int i = 0; i < SIMULATION_ITERATIONS; i++) {

        simulation_step_kernel <<<gridDim, blockDim>>> (d_read, d_write.heads);
        cudaDeviceSynchronize();

        double *tmp = d_write.heads;
        d_write.heads = d_read.heads;
        d_read.heads = tmp;

        if (i % STATISTICS_WRITE_FREQ == 0) {
            endTimer(&stepTimer);
            stats[i].stepTime = getElapsedTime(stepTimer);
            startTimer(&stepTimer);
        }
    }
}

int main(int argc, char *argv[]) {
    allocate_memory();
    init_read_ca();
    init_write_head();

    perform_simulation_on_GPU();

    if (WRITE_OUTPUT_TO_FILE){
        write_heads_to_file(d_write.heads, argv[0]);
    }

    if (WRITE_STATISTICS_TO_FILE) {
        write_statistics_to_file( argv[0]);
    }

    return 0;
}
