cmake_minimum_required(VERSION 3.16)

project(saturated-flow CXX)

find_package(CUDA REQUIRED)
cuda_select_nvcc_arch_flags(ARCH_FLAGS)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_library(common STATIC
        params.h
        parallel/common/cuda_error_check.cuh
        parallel/common/file_helper.h
        parallel/common/file_helper.cpp
        parallel/common/timer.h
        parallel/common/timer.cpp
        parallel/common/statistics.h
        parallel/common/statistics.cpp
        parallel/common/memory_management.cuh
        parallel/common/memory_management.cu
        parallel/kernels/iteration_step.cu
        )
set_target_properties(common PROPERTIES LINKER_LANGUAGE CUDA)

# global memory target
add_executable(g_standard parallel/global_memory/g_standard.cu)
target_link_libraries(g_standard common)
add_executable(g_standard_stress parallel/global_memory/g_standard_stress.cu)
target_link_libraries(g_standard_stress common)
add_executable(g_shared parallel/global_memory/g_shared.cu)
target_link_libraries(g_shared common)
add_executable(g_shared_stress parallel/global_memory/g_shared_stress.cu)
target_link_libraries(g_shared_stress common)
add_executable(g_hybrid parallel/global_memory/g_hybrid.cu)
target_link_libraries(g_hybrid common)
add_executable(g_hybrid_stress parallel/global_memory/g_hybrid_stress.cu)
target_link_libraries(g_hybrid_stress common)

# unified_sc memory target
add_executable(u_standard parallel/unified_memory/u_standard.cu)
target_link_libraries(u_standard common)
add_executable(u_standard_stress parallel/unified_memory/u_standard_stress.cu)
target_link_libraries(u_standard_stress common)
add_executable(u_shared parallel/unified_memory/u_shared.cu)
target_link_libraries(u_shared common)
add_executable(u_shared_stress parallel/unified_memory/u_shared_stress.cu)
target_link_libraries(u_shared_stress common)
add_executable(u_hybrid parallel/unified_memory/u_hybrid.cu)
target_link_libraries(u_hybrid common)
add_executable(u_hybrid_stress parallel/unified_memory/u_hybrid_stress.cu)
target_link_libraries(u_hybrid_stress common)

############################
### ACTIVE CELLS TARGETS ###
############################

## UNIFIED MEMORY ##

# atomic
add_executable(ac_u_atomic parallel/active_cells/unified_memory/atomic/ac_standard.cu)
target_link_libraries(ac_u_atomic common)
add_executable(ac_u_atomic_stress parallel/active_cells/unified_memory/atomic/ac_standard_stress.cu)
target_link_libraries(ac_u_atomic_stress common)

# naive
add_executable(ac_u_naive parallel/active_cells/unified_memory/naive/ac_standard.cu)
target_link_libraries(ac_u_naive common)
add_executable(ac_u_naive_stress parallel/active_cells/unified_memory/naive/ac_standard_stress.cu)
target_link_libraries(ac_u_naive_stress common)

# stream compaction
add_executable(ac_u_sc parallel/active_cells/unified_memory/sc/ac_standard.cu)
target_link_libraries(ac_u_sc common)
add_executable(ac_u_sc_stress parallel/active_cells/unified_memory/sc/ac_standard_stress.cu)
target_link_libraries(ac_u_sc_stress common)

# stream compaction adaptive
add_executable(ac_u_sc_adaptive parallel/active_cells/unified_memory/sc_adaptive/ac_standard.cu)
target_link_libraries(ac_u_sc_adaptive common)
add_executable(ac_u_sc_adaptive_stress parallel/active_cells/unified_memory/sc_adaptive/ac_standard_stress.cu)
target_link_libraries(ac_u_sc_adaptive_stress common)


## GLOBAL MEMORY ##

# atomic (deprecated)
add_executable(ac_g_atomic parallel/active_cells/global_memory/atomic/ac_standard.cu)
target_link_libraries(ac_g_atomic common)
add_executable(ac_g_atomic_stress parallel/active_cells/global_memory/atomic/ac_standard_stress.cu)
target_link_libraries(ac_g_atomic_stress common)

# naive
add_executable(ac_g_naive parallel/active_cells/global_memory/naive/ac_standard.cu)
target_link_libraries(ac_g_naive common)
add_executable(ac_g_naive_stress parallel/active_cells/global_memory/naive/ac_standard_stress.cu)
target_link_libraries(ac_g_naive_stress common)

# stream compaction
add_executable(ac_g_sc parallel/active_cells/global_memory/sc/ac_standard.cu)
target_link_libraries(ac_g_sc common)
add_executable(ac_g_sc_stress parallel/active_cells/global_memory/sc/ac_standard_stress.cu)
target_link_libraries(ac_g_sc_stress common)

# stream compaction adaptive
add_executable(ac_g_sc_adaptive parallel/active_cells/global_memory/sc_adaptive/ac_standard.cu)
target_link_libraries(ac_g_sc_adaptive common)
add_executable(ac_g_sc_adaptive_stress parallel/active_cells/global_memory/sc_adaptive/ac_standard_stress.cu)
target_link_libraries(ac_g_sc_adaptive_stress common)
