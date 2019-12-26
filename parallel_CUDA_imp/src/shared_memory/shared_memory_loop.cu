#include "shared_memory_common.h"

__global__ void simulation_step_kernel(struct CA *d_ca, double *d_write_head, int grid_size) {
    __shared__ double s_heads[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    __shared__ double s_K[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx_g = idx_y * COLS + idx_x;

    double Q, diff_head, tmp_t, ht1, ht2;

    if (idx_x < COLS && idx_y < ROWS) {
        unsigned x = threadIdx.x + 1;
        unsigned y = threadIdx.y + 1;

        s_heads[y][x] = d_ca->head[idx_g];
        s_K[y][x] = d_ca->K[idx_g];

        if (threadIdx.x == 0 && blockIdx.x != 0) // left
            s_heads[y][x - 1] = d_ca->head[idx_g - 1];
        if (threadIdx.x == BLOCK_SIZE - 1 && blockIdx.x != grid_size - 1) // right
            s_heads[y][x + 1] = d_ca->head[idx_g + 1];
        if (threadIdx.y == 0 && blockIdx.y != 0) // upper
            s_heads[y - 1][x] = d_ca->head[idx_g - COLS];
        if (threadIdx.y == BLOCK_SIZE - 1 && blockIdx.y != grid_size - 1) // bottom
            s_heads[y + 1][x] = d_ca->head[idx_g + COLS];
        __syncthreads();

        if (idx_y != 0 && idx_y != ROWS - 1) {
            for (int i = 0; i < KERNEL_LOOP_SIZE; i++) {
                Q = 0;
                if (idx_x >= 1) { // left neighbor
                    diff_head = s_heads[y][x - 1] - s_heads[y][x];
                    tmp_t = s_K[y][x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_y >= 1) { // upper neighbor
                    diff_head = s_heads[y - 1][x] - s_heads[y][x];
                    tmp_t = s_K[y][x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_x + 1 < COLS) { // right neighbor
                    diff_head = s_heads[y][x + 1] - s_heads[y][x];
                    tmp_t = s_K[y][x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
                if (idx_y + 1 < ROWS) { // bottom neighbor
                    diff_head = s_heads[y + 1][x] - s_heads[y][x];
                    tmp_t = s_K[y][x] * THICKNESS;
                    Q += diff_head * tmp_t;
                }
            }

            Q -= d_ca->Source[idx_g];
            ht1 = Q * DELTA_T;
            ht2 = AREA * d_ca->Sy[idx_g];

            d_write_head[idx_g] = s_heads[y][x] + ht1 / ht2;
        }
    }
}

void perform_simulation_on_GPU() {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    double gridSize = ceil(sqrt(blockCount));
    dim3 gridDim(gridSize, gridSize);

    for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
        simulation_step_kernel <<< gridDim, blockDim >>> (d_read_ca, d_write_head, gridSize);

        cudaDeviceSynchronize();

        double *tmp1 = d_write_head;
        CUDASAFECALL(
                cudaMemcpy(&d_write_head, &(d_read_ca->head), sizeof(d_read_ca->head), cudaMemcpyDeviceToHost));
        CUDASAFECALL(cudaMemcpy(&(d_read_ca->head), &tmp1, sizeof(tmp1), cudaMemcpyHostToDevice));
    }
}

int main(void) {
    init_host_ca();

    copy_data_from_CPU_to_GPU();

    perform_simulation_on_GPU();

    copy_data_from_GPU_to_CPU();

    return 0;
}
