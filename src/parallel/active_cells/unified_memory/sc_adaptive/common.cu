#include <thrust/device_vector.h>
#include "../../../common/memory_management.cuh"
#include "../../../common/statistics.h"
#include "../../../kernels/transition_kernels.cu"
#include "../../../kernels/dummy_kernels.cu"
#include "../../../kernels/ac_kernels.cu"
#include "../../utils.cu"


int main(int argc, char *argv[])
{
    auto h_ca = new CA();
    double *headsWrite;
    allocateManagedMemory(h_ca, headsWrite);
    initializeCA(h_ca);
    memcpy(headsWrite, h_ca->heads, sizeof(double) * ROWS * COLS);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDims = calculate_grid_dim();

    thrust::device_vector<int> activeCellsMask(ROWS * COLS, -1);
    thrust::device_vector<int> activeCellsIds(ROWS * COLS, -1);

    for (size_t i{0}; i < ROWS * COLS; ++i)
    {
        if (h_ca->sources[i] != 0)
        {
            activeCellsMask[i] = i;
        }
    }

    auto standardIterationTime = ac_utils::measure_standard_iteration_time(h_ca, headsWrite);

    bool isWholeGridActive = false;
    int devActiveCellsCount;
    int acIterCounter{};

    std::vector<StatPoint> stats;
    Timer stepTimer;
    stepTimer.start();
    for (int i{}; i < SIMULATION_ITERATIONS; ++i)
    {
        if (!isWholeGridActive)
        {
            thrust::copy_if(thrust::device, activeCellsMask.begin(), activeCellsMask.end(),
                            activeCellsIds.begin(), is_not_minus_one<int>());
            devActiveCellsCount = thrust::count_if(activeCellsIds.begin(), activeCellsIds.end(),
                                                   is_not_minus_one<int>());

            dim3 activeGridDim = calculate_grid_dim(devActiveCellsCount);

            ac_kernels::sc <<< activeGridDim, blockSize >>>(
                    *h_ca, headsWrite, thrust::raw_pointer_cast(&activeCellsIds[0]),
                    thrust::raw_pointer_cast(&activeCellsMask[0]),
                    devActiveCellsCount);

            for (int j = 0; j < EXTRA_KERNELS; j++)
            {
                dummy_kernels::dummy_active_sc <<< activeGridDim, blockSize >>>(
                        *h_ca, headsWrite, thrust::raw_pointer_cast(&activeCellsIds[0]),
                        devActiveCellsCount);
            }

            isWholeGridActive = devActiveCellsCount == ROWS * COLS;
            if (acIterCounter > 5)
            {
                isWholeGridActive = true;
                devActiveCellsCount = ROWS * COLS;
            }
        }
        else
        {
            kernels::standard_step <<< gridDims, blockSize >>>(*h_ca, headsWrite);
            for (int j = 0; j < EXTRA_KERNELS; j++)
            {
                dummy_kernels::dummy_active_naive <<< gridDims, blockSize >>>(*h_ca, headsWrite);
            }
        }
        ERROR_CHECK(cudaDeviceSynchronize());

        std::swap(h_ca->heads, headsWrite);
        save_step_stats(stats, stepTimer, devActiveCellsCount);

        if (i % STATISTICS_WRITE_FREQ == STATISTICS_WRITE_FREQ - 1)
        {
            if (stepTimer.elapsedNanoseconds() / STATISTICS_WRITE_FREQ >= standardIterationTime){
                acIterCounter++;
            }else{
                acIterCounter = 0;
            }
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
