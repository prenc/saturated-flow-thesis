#ifndef SATURATED_FLOW_THESIS_MEMORY_MANAGEMENT
#define SATURATED_FLOW_THESIS_MEMORY_MANAGEMENT

#include "cuda_error_check.cuh"
#include "../../params.h"
extern struct CA h_ca;

extern double *d_write_head;
extern CA *d_read_ca;

void copy_data_from_CPU_to_GPU();
void init_host_ca();
void copy_data_from_GPU_to_CPU();

#endif //SATURATED_FLOW_THESIS_MEMORY_MANAGEMENT
