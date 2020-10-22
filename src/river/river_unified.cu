#include "../parallel/common/unified_memory_management.cuh"
#include "../parallel/common/statistics.h"

double river_heads[] = RIVER_HEADS;

__global__ void simulation_step_kernel(struct CA d_ca, double *d_write_head, double river_head)
{
    unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned idx_g = idx_y * COLS + idx_x;

    double Q, diff_head, tmp_t, ht1, ht2;
    if (idx_x < COLS && idx_y < ROWS)
    {
        if (idx_y != 0 && idx_y != ROWS - 1)
        {
            Q = 0;
            if (idx_x >= 1)
            {
                diff_head = d_ca.heads[idx_g - 1] - d_ca.heads[idx_g];
                tmp_t = d_ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_y >= 1)
            {
                diff_head = d_ca.heads[(idx_y - 1) * COLS + idx_x] - d_ca.heads[idx_g];
                tmp_t = d_ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_x + 1 < COLS)
            {
                diff_head = d_ca.heads[idx_g + 1] - d_ca.heads[idx_g];
                tmp_t = d_ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_y + 1 < ROWS)
            {
                diff_head = d_ca.heads[(idx_y + 1) * COLS + idx_x] - d_ca.heads[idx_g];
                tmp_t = d_ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }

            if (idx_y == RIVER_POSITION)
            {
                double first_term_Q = (KSB * CELL_SIZE_X * W) / M;
                if (d_ca.heads[idx_g] > RIVER_BOTTOM)
                {
                    Q += first_term_Q * (river_head - d_ca.heads[idx_g]);
                } else
                {
                    Q += first_term_Q * (river_head - RIVER_BOTTOM + M);
                }
            }

            Q -= d_ca.sources[idx_g];

            ht1 = Q * DELTA_T;
            ht2 = AREA * d_ca.Sy[idx_g];
            d_write_head[idx_g] = d_ca.heads[idx_g] + ht1 / ht2;
        }
    }
}

void perform_simulation_on_GPU()
{
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    double gridSize = ceil(sqrt(blockCount));
    dim3 gridDim(gridSize, gridSize);

    int day_counter, steps_in_current_day = 0;
    double river_head;
    for (int i = 0; i < SIMULATION_STEPS; i++ && steps_in_current_day++)
    {
        river_head = river_heads[day_counter];

        simulation_step_kernel << < gridDim, blockDim >> > (d_read, d_write.heads, river_head);

        bool is_new_day = steps_in_current_day * DELTA_T >= SECONDS_IN_DAY;
        if (is_new_day)
        {
            day_counter++;
            is_new_day = false;
            steps_in_current_day = 0;

            if (WRITE_OUTPUT_TO_FILE)
            {
                write_river_heads_to_file(d_write.heads, river_head, day_counter);
            }
        }
        cudaDeviceSynchronize();

        double *tmp1 = d_write.heads;
        d_write.heads = d_read.heads;
        d_read.heads = tmp1;
    }
    if (WRITE_OUTPUT_TO_FILE)
    {
        day_counter++;
        write_river_heads_to_file(d_write.heads, river_head, day_counter);
    }
}

int main()
{
    allocate_memory();
    init_read_ca();
    init_write_head();

    perform_simulation_on_GPU();

    return 0;
}
