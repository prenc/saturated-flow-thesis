
#ifndef SATURATED_FLOW_THESIS_UNIFIED_MEMORY_MANAGEMENT_CUH
#define SATURATED_FLOW_THESIS_UNIFIED_MEMORY_MANAGEMENT_CUH
#include "../../params.h"
#include "cuda_error_check.cuh"
#include <iostream>
#include <numeric>
static struct CA d_read, d_write;

void allocate_memory();
void init_read_ca();
void init_write_head();
void free_allocated_memory();

#endif //SATURATED_FLOW_THESIS_UNIFIED_MEMORY_MANAGEMENT_CUH
