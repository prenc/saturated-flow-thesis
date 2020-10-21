#include "../common/memory_management.cuh"
#include "../common/statistics.h"

__global__ void simulation_step_kernel(struct CA ca, double *heads_write)
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
                diff_head = ca.heads[idx_g - 1] - ca.heads[idx_g];
                tmp_t = ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_y >= 1)
            {
                diff_head = ca.heads[(idx_y - 1) * COLS + idx_x] - ca.heads[idx_g];
                tmp_t = ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_x + 1 < COLS)
            {
                diff_head = ca.heads[idx_g + 1] - ca.heads[idx_g];
                tmp_t = ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }
            if (idx_y + 1 < ROWS)
            {
                diff_head = ca.heads[(idx_y + 1) * COLS + idx_x] - ca.heads[idx_g];
                tmp_t = ca.K[idx_g] * THICKNESS;
                Q += diff_head * tmp_t;
            }

            Q -= ca.sources[idx_g];
            ht1 = Q * DELTA_T;
            ht2 = AREA * ca.Sy[idx_g];

            heads_write[idx_g] = ca.heads[idx_g] + ht1 / ht2;
            if (heads_write[idx_g] < 0)
            {
                heads_write[idx_g] = 0;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    CA *h_ca = initializeCA();
    CA *d_ca = new CA();
    double *headsWrite;

    allocateMemory(d_ca, headsWrite);
    copyDataFromCpuToGpu(h_ca, d_ca);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((double) (ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    int gridSize = ceil(sqrt(blockCount));
    dim3 gridDims(gridSize, gridSize);

    Timer stepTimer{};
    startTimer(&stepTimer);

    for (unsigned i{}; i < SIMULATION_ITERATIONS; ++i)
    {
        simulation_step_kernel <<< gridDims, blockSize >>>(*d_ca, headsWrite);
        cudaDeviceSynchronize();

        double *tmpHeads = d_ca->heads;
        d_ca->heads = headsWrite;
        headsWrite = tmpHeads;

        if (i % STATISTICS_WRITE_FREQ == 0)
        {
            endTimer(&stepTimer);
            stats[i].stepTime = getElapsedTime(stepTimer);
            startTimer(&stepTimer);
        }
    }

    if (WRITE_OUTPUT_TO_FILE)
    {
        copyDataFromGpuToCpu(h_ca, d_ca);
        write_heads_to_file(h_ca->heads, argv[0]);
    }

    if (WRITE_STATISTICS_TO_FILE)
    {
        write_statistics_to_file(argv[0]);
    }

    free_allocated_memory(d_ca, headsWrite);
}
