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

    thrust::device_vector<int> activeCellsMask(ROWS * COLS, -1);
    thrust::device_vector<int> activeCellsIds(ROWS * COLS, -1);

    ac_utils::mark_sources_as_active_cells(h_ca, thrust::raw_pointer_cast(&activeCellsIds[0]));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDims = calculate_grid_dim();

    std::vector<StatPoint> stats;
    Timer stepTimer, activeCellsEvalTimer, transitionTimer;
    stepTimer.start();

    bool isWholeGridActive = false;
    dim3 *simulationGridDims;
    int devActiveCellsCount;

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

            simulationGridDims = &activeGridDim;
	        transitionTimer.start();
            ac_kernels::sc <<< *simulationGridDims, blockSize >>>(
			        *h_ca, headsWrite, thrust::raw_pointer_cast(&activeCellsIds[0]),
			        thrust::raw_pointer_cast(&activeCellsMask[0]),
			        devActiveCellsCount);

            for (int j = 0; j < EXTRA_KERNELS; j++)
            {
                dummy_kernels::dummy_active_sc <<< *simulationGridDims, blockSize >>>(
                            *h_ca,
                            headsWrite,
                            thrust::raw_pointer_cast(&activeCellsIds[0]),
                            devActiveCellsCount
                        );
            }
        }
        else
        {
	        transitionTimer.start();
	        kernels::standard_step <<< gridDims, blockSize >>>(*h_ca, headsWrite);

            for (int j = 0; j < EXTRA_KERNELS; j++)
            {
                dummy_kernels::dummy_active_naive <<< gridDims, blockSize >>>(*h_ca, headsWrite);
            }
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