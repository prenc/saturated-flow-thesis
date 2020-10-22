#ifndef SATURATED_FLOW_ITERATION_STEP_CUH
#define SATURATED_FLOW_ITERATION_STEP_CUH

#include "../common/memory_management.cuh"

namespace kernels
{
    __global__ void standard_step(struct CA ca, double *headsWrite);
    __global__ void hybrid_step(struct CA d_ca, double *headsWrite);
    __global__ void shared_step(struct CA d_ca, double *d_write_head);
}


#endif //SATURATED_FLOW_ITERATION_STEP_CUH
