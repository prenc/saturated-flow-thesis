#include "../kernels/transition_kernels.cu"
#include "../common/statistics.h"

int main(int argc, char *argv[])
{
    auto h_ca = new CA();
    double *headsWrite;
    allocateManagedMemory(h_ca, headsWrite);

    initializeCA(h_ca);
    memcpy(headsWrite, h_ca->heads, sizeof(double) * ROWS * COLS);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    const int blockCount = ceil((double) (ROWS * COLS) / (BLOCK_SIZE * BLOCK_SIZE));
    int gridSize = ceil(sqrt(blockCount));
    dim3 gridDims(gridSize, gridSize);

    std::vector<StatPoint> stats;
    Timer stepTimer, transitionTimer;
    stepTimer.start();

    for (unsigned i{}; i < SIMULATION_ITERATIONS; ++i)
    {

#ifdef STANDARD
        kernels::standard_step <<< gridDims, blockSize >>>(*h_ca, headsWrite);
#endif
#ifdef HYBRID
        kernels::hybrid_step <<< gridDims, blockSize >>>(*h_ca, headsWrite);
#endif
#ifdef SHARED
        kernels::shared_step <<< gridDims, blockSize >>>(*h_ca, headsWrite);
#endif
        ERROR_CHECK(cudaDeviceSynchronize());


        auto tmpHeads = h_ca->heads;
        h_ca->heads = headsWrite;
        headsWrite = tmpHeads;

        if (i % STATISTICS_WRITE_FREQ == STATISTICS_WRITE_FREQ - 1)
        {
            stepTimer.stop();
            auto stat = new StatPoint(
                    -1,
                    stepTimer.elapsedNanoseconds);
            stats.push_back(*stat);
            stepTimer.start();
        }
    }

    if (WRITE_OUTPUT_TO_FILE)
    {
        saveHeadsInFile(h_ca->heads, argv[0]);
    }

    if (WRITE_STATISTICS_TO_FILE)
    {
        writeStatisticsToFile(stats, argv[0]);
    }

    freeAllocatedMemory(h_ca, headsWrite);
}
