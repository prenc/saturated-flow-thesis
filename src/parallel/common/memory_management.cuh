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

dim3 calculate_grid_dim();

dim3 calculate_grid_dim(int cell_count);

void save_output_and_free_memory(char *argv[], struct CA *h_ca, struct CA *d_ca, double *headsWrite, std::vector<StatPoint> &stats)
void save_output_and_free_memory(char *argv[], struct CA *h_ca, double *headsWrite, std::vector<StatPoint> &stats)

#endif //SATURATED_FLOW_THESIS_MEMORY_MANAGEMENT
