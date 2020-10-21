#ifndef SATURATED_FLOW_THESIS_MEMORY_MANAGEMENT
#define SATURATED_FLOW_THESIS_MEMORY_MANAGEMENT

#include "cuda_error_check.cuh"
#include "../../params.h"

struct CA
{
    double *heads;
    double *Sy;
    double *K;
    double *sources;
};

CA *initializeCA();

void allocateManagedMemory(CA *&ca, double *&heads_write);

void allocateMemory(CA *&ca, double *&heads_write);

void copyDataFromCpuToGpu(CA *&h_ca, CA *&d_ca);

void copyDataFromGpuToCpu(CA *&h_ca, CA *&d_ca);

void free_allocated_memory(CA *&d_ca, double *&headsWrite);

#endif //SATURATED_FLOW_THESIS_MEMORY_MANAGEMENT
