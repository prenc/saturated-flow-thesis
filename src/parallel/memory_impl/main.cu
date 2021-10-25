#include "../kernels/transition_kernels.cu"
#include "../kernels/dummy_kernels.cu"
#include "../utils/statistics.h"

int main(int argc, char *argv[])
{
    auto d_ca = new CA();
    double *headsWrite;

#ifdef GLOBAL
    CA *h_ca = new CA();

    h_ca->heads = new double[ROWS * COLS]();
    h_ca->Sy = new double[ROWS * COLS]();
    h_ca->K = new double[ROWS * COLS]();
    h_ca->sources = new double[ROWS * COLS]();

    initializeCA(h_ca);

    allocateMemory(d_ca, headsWrite);
    copyDataFromCpuToGpu(h_ca, d_ca, headsWrite);
#else
    allocateManagedMemory(d_ca, headsWrite);
    initializeCA(d_ca);
#endif

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDims = calculate_grid_dim();

    std::vector<StatPoint> stats;
    Timer stepTimer, transitionTimer;
    stepTimer.start();

    for (unsigned i{}; i < SIMULATION_ITERATIONS; ++i)
    {

#ifdef STANDARD
        kernels::standard_step <<< gridDims, blockSize >>>(*d_ca, headsWrite);
#elif HYBRID
        kernels::hybrid_step <<< gridDims, blockSize >>>(*d_ca, headsWrite);
#elif SHARED
        kernels::shared_step <<< gridDims, blockSize >>>(*d_ca, headsWrite);
#endif
        for (int j = 0; j < EXTRA_KERNELS; j++)
        {
            dummy_kernels::dummy_all <<< gridDims, blockSize >>>(*d_ca, headsWrite);
        }
        ERROR_CHECK(cudaDeviceSynchronize());

        std::swap(d_ca->heads, headsWrite);
        save_step_stats(stats, &stepTimer, i);
    }

#ifdef GLOBAL
    save_output_and_free_memory(argv, h_ca, d_ca, headsWrite, stats);
#else
    save_output_and_free_memory(argv, d_ca, headsWrite, stats);
#endif
}
