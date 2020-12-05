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

void initializeCA(CA *&ca);

void allocateManagedMemory(CA *&ca, double *&heads_write);

void allocateMemory(CA *&ca, double *&heads_write);

void copyDataFromCpuToGpu(CA *&h_ca, CA *&d_ca, double *headWrite);

void copyDataFromGpuToCpu(CA *&h_ca, CA *&d_ca);

void freeAllocatedMemory(CA *&d_ca, double *&headsWrite);

#endif //SATURATED_FLOW_THESIS_MEMORY_MANAGEMENT
