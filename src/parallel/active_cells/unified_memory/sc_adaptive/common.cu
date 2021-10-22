#include <thrust/device_vector.h>
#include <algorithm>
#include "../../../common/memory_management.cuh"
#include "../../../common/statistics.h"
#include "../../../kernels/transition_kernels.cu"
#include "../../../kernels/dummy_kernels.cu"
#include "../../../kernels/ac_kernels.cu"


int main(int argc, char *argv[])
{
    auto h_ca = new CA();
    double *headsWrite;
    allocateManagedMemory(h_ca, headsWrite);

    thrust::device_vector<int> activeCellsMask(ROWS * COLS, -1);
    thrust::device_vector<int> activeCellsIds(ROWS * COLS, -1);

    initializeCA(h_ca);
    memcpy(headsWrite, h_ca->heads, sizeof(double) * ROWS * COLS);
    for (size_t i{0}; i < ROWS * COLS; ++i)
    {
        if (h_ca->sources[i] != 0)
        {
            activeCellsMask[i] = i;
        }
    }

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDims = calculate_grid_dim();

    std::vector<StatPoint> stats;
    Timer stepTimer, activeCellsEvalTimer, transitionTimer;

    std::vector<size_t> times{};
    for (size_t i{}; i < 5; ++i)
    {
        stepTimer.start();
        kernels::standard_step <<< gridDims, blockSize >>>(*h_ca, headsWrite);
        ERROR_CHECK(cudaDeviceSynchronize());
        stepTimer.stop();
        times.push_back(stepTimer.elapsedNanoseconds());
    }
    std::sort(times.begin(), times.end());
    auto standardIterationTime = times[0];

    bool isWholeGridActive = false;
    int devActiveCellsCount;
    int acIterCounter{};
    stepTimer.start();
    for (int i{}; i < SIMULATION_ITERATIONS; ++i)
    {
        if (!isWholeGridActive)
        {
            activeCellsEvalTimer.start();
            thrust::copy_if(thrust::device, activeCellsMask.begin(), activeCellsMask.end(),
                            activeCellsIds.begin(), is_not_minus_one<int>());
            devActiveCellsCount = thrust::count_if(activeCellsIds.begin(), activeCellsIds.end(),
                                                   is_not_minus_one<int>());
            activeCellsEvalTimer.stop();

            isWholeGridActive = devActiveCellsCount == ROWS * COLS;

            dim3 activeGridDim = calculate_grid_dim(devActiveCellsCount);

            transitionTimer.start();
            ac_kernels::sc <<< activeGridDim, blockSize >>>(
                    *h_ca, headsWrite, thrust::raw_pointer_cast(&activeCellsIds[0]),
                    thrust::raw_pointer_cast(&activeCellsMask[0]),
                    devActiveCellsCount);

#ifdef EXTRA_KERNELS
            for (int j = 0; j < EXTRA_KERNELS; j++)
            {
                dummy_kernels::dummy_active_sc <<< activeGridDim, blockSize >>>(
                        *h_ca, headsWrite, thrust::raw_pointer_cast(&activeCellsIds[0]),
                        devActiveCellsCount);
            }
#endif

            if (acIterCounter > 5)
            {
                isWholeGridActive = true;
                devActiveCellsCount = ROWS * COLS;
                activeCellsEvalTimer.start();
                activeCellsEvalTimer.stop();
            }
        }
        else
        {
            transitionTimer.start();
            kernels::standard_step <<< gridDims, blockSize >>>(*h_ca, headsWrite);
#ifdef EXTRA_KERNELS
            for (int j = 0; j < EXTRA_KERNELS; j++)
            {
                dummy_kernels::dummy_active_naive <<< gridDims, blockSize >>>(*h_ca, headsWrite);
            }
#endif
        }

        ERROR_CHECK(cudaDeviceSynchronize());
        transitionTimer.stop();

        std::swap(h_ca->heads, headsWrite);

        if (i % STATISTICS_WRITE_FREQ == STATISTICS_WRITE_FREQ - 1)
        {
            stepTimer.stop();
            auto stat = new StatPoint(
                    devActiveCellsCount / (double) (ROWS * COLS),
                    stepTimer.elapsedNanoseconds(),
                    transitionTimer.elapsedNanoseconds(),
                    activeCellsEvalTimer.elapsedNanoseconds());
            stat->adaptiveTime = standardIterationTime;
            if (stepTimer.elapsedNanoseconds() / STATISTICS_WRITE_FREQ >= standardIterationTime)
            {
                acIterCounter++;
            }
            else
            {
                acIterCounter = 0;
            }
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
