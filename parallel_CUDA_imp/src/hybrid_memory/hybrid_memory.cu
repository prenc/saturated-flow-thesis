#include "hybrid_memory_common.h"

__global__ void simulation_step_kernel(struct CA *d_ca, double *d_write_head) {
    __shared__ double s_heads[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_K[BLOCK_SIZE][BLOCK_SIZE];
    unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx_g = idx_y * COLS + idx_x;

    s_heads[threadIdx.y][threadIdx.x] = d_ca->head[idx_g];
    s_K[threadIdx.y][threadIdx.x] = d_ca->K[idx_g];

    __syncthreads();

    double Q, diff_head, tmp_t, ht1, ht2;
    if (idx_x < COLS && idx_y < ROWS)
        if (idx_y != 0 && idx_y != ROWS - 1) {
        	Q = 0;
            if (idx_x >= 1) { // left neighbor
                if (threadIdx.x >= 1)
                    diff_head = s_heads[threadIdx.y][threadIdx.x - 1] - s_heads[threadIdx.y][threadIdx.x];
                else
                    diff_head = d_ca->head[idx_g - 1] - s_heads[threadIdx.y][threadIdx.x];
                tmp_t = s_K[threadIdx.y][threadIdx.x] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_y >= 1) { // upper neighbor
                if (threadIdx.y >= 1)
                    diff_head = s_heads[threadIdx.y - 1][threadIdx.x] - s_heads[threadIdx.y][threadIdx.x];
                else
                    diff_head = d_ca->head[(idx_y - 1) * COLS + idx_x] - s_heads[threadIdx.y][threadIdx.x];
                tmp_t = s_K[threadIdx.y][threadIdx.x] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_x + 1 < COLS) { // right neighbor
                if (threadIdx.x < BLOCK_SIZE - 1)
                    diff_head = s_heads[threadIdx.y][threadIdx.x + 1] - s_heads[threadIdx.y][threadIdx.x];
                else
                    diff_head = d_ca->head[idx_g + 1] - s_heads[threadIdx.y][threadIdx.x];
                tmp_t = s_K[threadIdx.y][threadIdx.x] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_y + 1 < ROWS) { // bottom neighbor
                if (threadIdx.y < BLOCK_SIZE - 1)
                    diff_head = s_heads[threadIdx.y + 1][threadIdx.x] - s_heads[threadIdx.y][threadIdx.x];
                else
                    diff_head = d_ca->head[(idx_y + 1) * COLS + idx_x] - s_heads[threadIdx.y][threadIdx.x];
                tmp_t = s_K[threadIdx.y][threadIdx.x] * THICKNESS;
                Q += diff_head * tmp_t;
            }

            Q -= d_ca->Source[idx_g];

            ht1 = Q * DELTA_T;
            ht2 = AREA * d_ca->Sy[idx_g];

            d_write_head[idx_g] = s_heads[threadIdx.y][threadIdx.x] + ht1 / ht2;
        }
}

void perform_simulation_on_GPU() {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    double gridSize = ceil(sqrt(blockCount));
    dim3 gridDim(gridSize, gridSize);

    for (int i = 0; i < SIMULATION_ITERATIONS; i++) {
        simulation_step_kernel << < gridDim, blockDim >> > (d_read_ca, d_write_head);

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
