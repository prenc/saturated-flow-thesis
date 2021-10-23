#include "../../../common/memory_management.cuh"
#include "../../../common/statistics.h"
#include "../../../kernels/transition_kernels.cu"
#include "../../../kernels/utils.cu"
#include "../../../kernels/dummy_kernels.cu"
#include "../../../kernels/ac_kernels.cu"

int main(int argc, char *argv[])
{
    auto h_ca = new CA();
    double *headsWrite;
    allocateManagedMemory(h_ca, headsWrite);
    initializeCA(h_ca);
    memcpy(headsWrite, h_ca->heads, sizeof(double) * ROWS * COLS);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDims = calculate_grid_dim();

    std::vector<StatPoint> stats;
    Timer stepTimer;
    stepTimer.start();
    for (int i{}; i < SIMULATION_ITERATIONS; ++i)
    {
        ac_kernels::naive <<< gridDims, blockSize >>>(*h_ca, headsWrite);
        for (int j = 0; j < EXTRA_KERNELS; j++)
        {
            dummy_kernels::dummy_active_naive <<< gridDims, blockSize >>>(*h_ca, headsWrite);
        }
        ERROR_CHECK(cudaDeviceSynchronize());

        std::swap(h_ca->heads, headsWrite);
        save_step_stats(stats, stepTimer);
    }
    save_output_and_free_memory(argv, h_ca, headsWrite, stats);
}