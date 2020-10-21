#include "../common/memory_management.cuh"
#include "../common/statistics.h"
#include "../kernels/iteration_step.cuh"


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
#ifdef STANDARD
        kernels::standard_step <<< gridDims, blockSize >>>(*d_ca, headsWrite);
#endif
#ifdef HYBRID
        kernels::hybrid_step <<< gridDims, blockSize >>>(*d_ca, headsWrite);
#endif
#ifdef SHARED
        kernels::shared_step <<< gridDims, blockSize >>>(*d_ca, headsWrite, gridSize);
#endif
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
        writeHeadsToFile(h_ca->heads, argv[0]);
    }

    if (WRITE_STATISTICS_TO_FILE)
    {
        writeStatisticsToFile(argv[0]);
    }

    freeAllocatedMemory(d_ca, headsWrite);
}
