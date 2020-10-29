#include "../kernels/iteration_step.cu"
#include "../common/statistics.h"

int main(int argc, char *argv[])
{
    CA *d_ca = new CA();
    CA *h_ca = new CA();
    double *headsWrite;

    h_ca->heads = new double[ROWS * COLS]();
    h_ca->Sy = new double[ROWS * COLS]();
    h_ca->K = new double[ROWS * COLS]();
    h_ca->sources = new double[ROWS * COLS]();

    initializeCA(h_ca);

    allocateMemory(d_ca, headsWrite);
    copyDataFromCpuToGpu(h_ca, d_ca);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((double) (ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    int gridSize = ceil(sqrt(blockCount));
    dim3 gridDims(gridSize, gridSize);

    std::vector<StatPoint> stats;
    Timer stepTimer;
    stepTimer.start();

    for (unsigned i{}; i < SIMULATION_ITERATIONS; ++i)
    {
#ifdef STANDARD
        kernels::standard_step <<< gridDims, blockSize >>>(*d_ca, headsWrite);
#endif
#ifdef HYBRID
        kernels::hybrid_step <<< gridDims, blockSize >>>(*d_ca, headsWrite);
#endif
#ifdef SHARED
        kernels::shared_step <<< gridDims, blockSize >>>(*d_ca, headsWrite);
#endif
        ERROR_CHECK(cudaDeviceSynchronize());

        double *tmpHeads = d_ca->heads;
        d_ca->heads = headsWrite;
        headsWrite = tmpHeads;

        if (i % STATISTICS_WRITE_FREQ == 0)
        {
            stepTimer.stop();
            auto stat = new StatPoint();
            stat->stepTime = stepTimer.elapsedNanoseconds();
            stats.push_back(*stat);
            stepTimer.start();
        }
    }

    if (WRITE_OUTPUT_TO_FILE)
    {
        copyDataFromGpuToCpu(h_ca, d_ca);
        saveHeadsInFile(h_ca->heads, argv[0]);
    }

    if (WRITE_STATISTICS_TO_FILE)
    {
        writeStatisticsToFile(stats, argv[0]);
    }

    freeAllocatedMemory(d_ca, headsWrite);
}
