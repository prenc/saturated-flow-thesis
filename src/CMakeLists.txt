cmake_minimum_required(VERSION 3.16)

project(saturated-flow CXX)

find_package(CUDA REQUIRED)
cuda_select_nvcc_arch_flags(ARCH_FLAGS)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

add_library(common STATIC
        params.h
        parallel/utils/cuda_error_check.cuh
        parallel/utils/file_helper.h
        parallel/utils/file_helper.cpp
        parallel/utils/timer.h
        parallel/utils/timer.cpp
        parallel/utils/statistics.h
        parallel/utils/statistics.cpp
        parallel/utils/memory_management.cuh
        parallel/utils/memory_management.cu
        parallel/kernels/transition_kernels.cu
        parallel/active_cells_impl/utils.cu
        parallel/kernels/dummy_kernels.cu
        parallel/kernels/ac_kernels.cu
        parallel/active_cells_impl/sc/utils.cu
        )
set_target_properties(common PROPERTIES LINKER_LANGUAGE CUDA)

# global memory target
add_executable(g_standard parallel/memory_impl/variants/g_standard.cu)
target_link_libraries(g_standard common)
add_executable(g_standard_stress parallel/memory_impl/variants/g_standard_stress.cu)
target_link_libraries(g_standard_stress common)
add_executable(g_shared parallel/memory_impl/variants/g_shared.cu)
target_link_libraries(g_shared common)
add_executable(g_shared_stress parallel/memory_impl/variants/g_shared_stress.cu)
target_link_libraries(g_shared_stress common)
add_executable(g_hybrid parallel/memory_impl/variants/g_hybrid.cu)
target_link_libraries(g_hybrid common)
add_executable(g_hybrid_stress parallel/memory_impl/variants/g_hybrid_stress.cu)
target_link_libraries(g_hybrid_stress common)

# unified_sc memory target
add_executable(u_standard parallel/memory_impl/variants/u_standard.cu)
target_link_libraries(u_standard common)
add_executable(u_standard_stress parallel/memory_impl/variants/u_standard_stress.cu)
target_link_libraries(u_standard_stress common)
add_executable(u_shared parallel/memory_impl/variants/u_shared.cu)
target_link_libraries(u_shared common)
add_executable(u_shared_stress parallel/memory_impl/variants/u_shared_stress.cu)
target_link_libraries(u_shared_stress common)
add_executable(u_hybrid parallel/memory_impl/variants/u_hybrid.cu)
target_link_libraries(u_hybrid common)
add_executable(u_hybrid_stress parallel/memory_impl/variants/u_hybrid_stress.cu)
target_link_libraries(u_hybrid_stress common)

############################
### ACTIVE CELLS TARGETS ###
############################

## UNIFIED MEMORY ##

# naive
add_executable(ac_u_naive parallel/active_cells_impl/naive/variants/u_ac_standard.cu)
target_link_libraries(ac_u_naive common)
add_executable(ac_u_naive_stress parallel/active_cells_impl/naive/variants/u_standard_stress.cu)
target_link_libraries(ac_u_naive_stress common)

# stream compaction
add_executable(ac_u_sc parallel/active_cells_impl/sc/variants/u_ac_standard.cu)
target_link_libraries(ac_u_sc common)
add_executable(ac_u_sc_stress parallel/active_cells_impl/sc/variants/u_ac_standard_stress.cu)
target_link_libraries(ac_u_sc_stress common)

# stream compaction adaptive
add_executable(ac_u_sc_adaptive parallel/active_cells_impl/sc/variants/u_ac_adaptive.cu)
target_link_libraries(ac_u_sc_adaptive common)
add_executable(ac_u_sc_adaptive_stress parallel/active_cells_impl/sc/variants/u_ac_adaptive_stress.cu)
target_link_libraries(ac_u_sc_adaptive_stress common)

# stream compaction adaptive naive
add_executable(ac_u_sc_adaptive_naive parallel/active_cells_impl/sc/variants/u_ac_adaptive_naive.cu)
target_link_libraries(ac_u_sc_adaptive_naive common)

## GLOBAL MEMORY ##

# naive
add_executable(ac_g_naive parallel/active_cells_impl/naive/variants/g_ac_standard.cu)
target_link_libraries(ac_g_naive common)
add_executable(ac_g_naive_stress parallel/active_cells_impl/naive/variants/g_ac_standard_stress.cu)
target_link_libraries(ac_g_naive_stress common)

# stream compaction
add_executable(ac_g_sc parallel/active_cells_impl/sc/variants/g_ac_standard.cu)
target_link_libraries(ac_g_sc common)
add_executable(ac_g_sc_stress parallel/active_cells_impl/sc/variants/g_ac_standard_stress.cu)
target_link_libraries(ac_g_sc_stress common)

# stream compaction adaptive
add_executable(ac_g_sc_adaptive parallel/active_cells_impl/sc/variants/g_ac_adaptive.cu)
target_link_libraries(ac_g_sc_adaptive common)
add_executable(ac_g_sc_adaptive_stress parallel/active_cells_impl/sc/variants/g_ac_adaptive_stress.cu)
target_link_libraries(ac_g_sc_adaptive_stress common)
