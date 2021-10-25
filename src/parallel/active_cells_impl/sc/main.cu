#include <thrust/device_vector.h>
#include "../../utils/memory_management.cuh"
#include "../../utils/statistics.h"
#include "../../kernels/transition_kernels.cu"
#include "../../kernels/dummy_kernels.cu"
#include "../../kernels/ac_kernels.cu"
#include "utils.cu"

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
    memcpy(headsWrite, d_ca->heads, sizeof(double) * ROWS * COLS);
#endif

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDims = calculate_grid_dim();

    thrust::device_vector<int> activeCellsMask(ROWS * COLS, -1);
    thrust::device_vector<int> activeCellsIds(ROWS * COLS, -1);

    for (size_t i{0}; i < ROWS * COLS; ++i)
    {
#ifdef GLOBAL
        if (h_ca->sources[i] != 0)
#else
        if (d_ca->sources[i] != 0)
#endif
        {
            activeCellsMask[i] = i;
        }
    }

#ifdef ADAPTIVE
    auto standardIterationTime = sc_utils::measure_standard_iteration_time(d_ca, headsWrite);
    int sc_steps_with_higher_time_than_standard{};
    bool shouldAdapt = false;
#endif

    bool isWholeGridActive = false;
    int devActiveCellsCount;

    std::vector<StatPoint> stats;
    Timer stepTimer;

    stepTimer.start();
    for (int i{}; i < SIMULATION_ITERATIONS; ++i)
    {
#ifdef ADAPTIVE
        if (isWholeGridActive || shouldAdapt)
#else
        if (isWholeGridActive)
#endif
        {
#ifdef NAIVE
            ac_kernels::naive <<< gridDims, blockSize >>>(*d_ca, headsWrite);
#else
            kernels::standard_step <<< gridDims, blockSize >>>(*d_ca, headsWrite);
#endif
            for (int j = 0; j < EXTRA_KERNELS; j++)
            {
                dummy_kernels::dummy_active_naive <<< gridDims, blockSize >>>(*d_ca, headsWrite);
            }
        }
        else
        {
            thrust::copy_if(thrust::device, activeCellsMask.begin(), activeCellsMask.end(),
                            activeCellsIds.begin(), is_not_minus_one<int>());
            devActiveCellsCount = thrust::count_if(activeCellsIds.begin(), activeCellsIds.end(),
                                                   is_not_minus_one<int>());

            dim3 activeGridDim = calculate_grid_dim(devActiveCellsCount);

            ac_kernels::sc <<< activeGridDim, blockSize >>>(
                    *d_ca, headsWrite, thrust::raw_pointer_cast(&activeCellsIds[0]),
                            thrust::raw_pointer_cast(&activeCellsMask[0]),
                            devActiveCellsCount);

            for (int j = 0; j < EXTRA_KERNELS; j++)
            {
                dummy_kernels::dummy_active_sc <<< activeGridDim, blockSize >>>(
                        *d_ca, headsWrite, thrust::raw_pointer_cast(&activeCellsIds[0]),
                                devActiveCellsCount);
            }

            isWholeGridActive = devActiveCellsCount == ROWS * COLS;
        }
        ERROR_CHECK(cudaDeviceSynchronize());

        std::swap(d_ca->heads, headsWrite);
        save_step_stats(stats, &stepTimer, i, devActiveCellsCount);

#ifdef ADAPTIVE
        sc_utils::set_sc_counter(stepTimer, i, &sc_steps_with_higher_time_than_standard, standardIterationTime);
        shouldAdapt = sc_utils::check_if_model_should_adapt(
                sc_steps_with_higher_time_than_standard,
                &devActiveCellsCount);
#endif
    }

#ifdef GLOBAL
    save_output_and_free_memory(argv, h_ca, d_ca, headsWrite, stats);
#else
    save_output_and_free_memory(argv, d_ca, headsWrite, stats);
#endif
}
