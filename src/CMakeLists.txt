cmake_minimum_required(VERSION 3.17)

project(saturated-flow CXX)

find_package(CUDA REQUIRED)
cuda_select_nvcc_arch_flags(ARCH_FLAGS)

enable_language(CUDA)

file(GLOB COMMON
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
        parallel/kernels/iteration_step.cuh
        )

add_executable(standard parallel/global_memory/standard.cu ${COMMON})
add_executable(standard_loop parallel/global_memory/standard_loop.cu ${COMMON})
add_executable(shared parallel/global_memory/shared.cu ${COMMON})
add_executable(shared_loop parallel/global_memory/shared_loop.cu ${COMMON})
add_executable(hybrid parallel/global_memory/hybrid.cu ${COMMON})
add_executable(hybrid_loop parallel/global_memory/hybrid_loop.cu ${COMMON})

add_executable(global_active_cells parallel/active_cells/global/global_active_cells.cu ${COMMON})
add_executable(global_active_cells_loop parallel/active_cells/global/global_active_cells_loop.cu ${COMMON})
add_executable(unified_memory_loop parallel/unified_memory/unified_memory_loop.cu ${COMMON})
add_executable(unified_memory parallel/unified_memory/unified_memory.cu ${COMMON})
add_executable(active_cells parallel/active_cells/unified/unified_active_cells.cu ${COMMON})
add_executable(active_cells_loop parallel/active_cells/unified/unified_active_cells_loop.cu ${COMMON})
add_executable(river river/river_unified.cu ${COMMON})
