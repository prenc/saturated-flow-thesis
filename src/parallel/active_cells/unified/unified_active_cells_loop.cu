#include "../../unified_memory/unified_memory_common.h"

__device__ int active_cells_idx[ROWS * COLS];

__managed__ int dev_active_cells_count = 0;

__device__ void my_push_back(int cellIdx) {
    int insert_ptr = atomicAdd(&dev_active_cells_count, 1);
    active_cells_idx[insert_ptr] = cellIdx;
}

__global__ void simulation_step_kernel(struct CA d_ca, double *d_write_head) {
    int activeBlockCount = ceil((double) dev_active_cells_count / (BLOCK_SIZE * BLOCK_SIZE));
    int activeGridSize = ceil(sqrtf(activeBlockCount));

    unsigned ac_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned ac_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned ac_idx_g = ac_idx_y * blockDim.x * activeGridSize + ac_idx_x;

    double Q, diff_head, tmp_t, ht1, ht2;
    for (int i = 0; i < KERNEL_LOOP_SIZE; i++) {
        if (ac_idx_g < dev_active_cells_count) {
            unsigned idx_g = active_cells_idx[ac_idx_g];
            unsigned idx_x = idx_g % COLS;
            unsigned idx_y = idx_g / COLS;

            Q = 0;
            if (idx_y != 0 && idx_y != ROWS - 1) {
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

            ht1  = Q * DELTA_T;
            ht2 = AREA * d_ca.Sy[idx_g];

            d_write_head[idx_g] = d_ca.head[idx_g] + ht1 / ht2;
            if (d_write_head[idx_g] < 0) {
                d_write_head[idx_g] = 0;
            }
        }
    }
}

__global__ void find_active_cells_kernel(struct CA d_ca) {
    unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx_g = idx_y * COLS + idx_x;

    if (idx_x < ROWS && idx_y < COLS) {
        if (d_ca.head[idx_g] < headFixed || d_ca.Source[idx_g] != 0) {
            my_push_back(idx_g);
            return;
        }
        if (idx_x >= 1) {
            if (d_ca.head[idx_g - 1] < headFixed) {
                my_push_back(idx_g);
                return;
            }
        }
        if (idx_y >= 1) {
            if (d_ca.head[idx_g - COLS] < headFixed) {
                my_push_back(idx_g);
                return;
            }
        }
        if (idx_x + 1 < COLS) {
            if (d_ca.head[idx_g + 1] < headFixed) {
                my_push_back(idx_g);
                return;
            }
        }
        if (idx_y + 1 < ROWS) {
            if (d_ca.head[idx_g + COLS] < headFixed) {
                my_push_back(idx_g);
                return;
            }
        }
    }
}

void perform_simulation_on_GPU() {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    const int blockCount = ceil((ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    int gridSize = ceil(sqrt(blockCount));
    dim3 gridDim(gridSize, gridSize);

    int activeBlockCount, activeGridSize;

    Timer stepTimer;

    startTimer(&stepTimer);
    bool isWholeGridActive = false;

    for (int i = 0; i < SIMULATION_ITERATIONS; i++) {

        dim3 *simulationGridDim;
        if (!isWholeGridActive) {

            dev_active_cells_count = 0;
            find_active_cells_kernel <<<gridDim, blockSize>>>(d_read);
            cudaDeviceSynchronize();

            isWholeGridActive = dev_active_cells_count == ROWS*COLS;

            activeBlockCount = ceil((double) dev_active_cells_count / (BLOCK_SIZE * BLOCK_SIZE));
            activeGridSize = ceil(sqrt(activeBlockCount));
            dim3 activeGridDim(activeGridSize, activeGridSize);

            simulationGridDim = &activeGridDim;
        } else {
            simulationGridDim = &gridDim;
        }

        simulation_step_kernel <<<*simulationGridDim, blockSize>>>(d_read, d_write.head);
        cudaDeviceSynchronize();

        double *tmp = d_write.head;
        d_write.head = d_read.head;
        d_read.head = tmp;

        if (i % STATISTICS_WRITE_FREQ == 0) {
            endTimer(&stepTimer);
	        stats[i].coverage = double(dev_active_cells_count) / (ROWS * COLS);
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

    if (WRITE_OUTPUT_TO_FILE) {
        write_heads_to_file(d_write.head, argv[0]);
    }

    if (WRITE_STATISTICS_TO_FILE) {
        write_statistics_to_file(stats, argv[0]);
    }

    return 0;
}









