#ifndef AC_UTILS
#define AC_UTILS

#include <thrust/device_vector.h>
#include <algorithm>
#include "../../common/memory_management.cuh"
#include "../../kernels/transition_kernels.cu"
#include "../../common/statistics.h"

namespace ac_utils {
    size_t measure_standard_iteration_time(struct CA *h_ca, double *headsWrite){
        Timer stepTimer;
        std::vector<size_t> times{};
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDims = calculate_grid_dim();

        for (size_t i{}; i < 5; ++i)
        {
            stepTimer.start();
            kernels::standard_step <<< gridDims, blockSize >>>(*h_ca, headsWrite);
            ERROR_CHECK(cudaDeviceSynchronize());
            stepTimer.stop();
            times.push_back(stepTimer.elapsedNanoseconds);
        }
        std::sort(times.begin(), times.end());
        return times[0];
    }

    void set_sc_counter(Timer stepTimer, int stepCounter, int *sc_steps_with_higher_time_than_standard,
                        double standardIterationTime){
        if (stepCounter % STATISTICS_WRITE_FREQ == STATISTICS_WRITE_FREQ - 1)
        {
            if (stepTimer.elapsedNanoseconds / STATISTICS_WRITE_FREQ >= standardIterationTime){
                *sc_steps_with_higher_time_than_standard += 1;
            }else{
                *sc_steps_with_higher_time_than_standard = 0;
            }
        }
    }
    bool check_if_model_should_adapt(int sc_steps_with_higher_time_than_standard, int *devActiveCellsCount){
        if (sc_steps_with_higher_time_than_standard > MINIMUM_SC_RUNS_WITH_HIGHER_TIME_TO_ADAPT)
        {
            *devActiveCellsCount = ROWS * COLS;
            return true;
        }
        return false;
    }

}

#endif
