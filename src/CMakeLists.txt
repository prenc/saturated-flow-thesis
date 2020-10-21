cmake_minimum_required(VERSION 3.16)

project(saturated_flow_thesis)
enable_language(CUDA)

file( GLOB COMMON
        params.h
        parallel/common/cuda_error_check.cuh
        parallel/common/file_helper.h
        parallel/common/file_helper.cpp
        parallel/common/timer.h
        parallel/common/timer.cpp
        parallel/common/statistics.h
        parallel/common/statistics.cpp
        )

file( GLOB NORMAL_MEMORY_MANAGEMENT
        parallel/common/memory_management.cuh
        parallel/common/memory_management.cu
        )

file( GLOB UNIFIED_MEMORY_MANAGEMENT
        parallel/common/unified_memory_management.cu
        parallel/common/unified_memory_management.cuh
        )

# NORMAL MEMORY MANAGEMENT
add_executable(global_memory parallel/global_memory/global_memory.cu ${COMMON} ${NORMAL_MEMORY_MANAGEMENT})
add_executable(global_memory_loop parallel/global_memory/global_memory_loop.cu ${COMMON} ${NORMAL_MEMORY_MANAGEMENT} )
add_executable(shared_memory parallel/shared_memory/shared_memory.cu ${COMMON} ${NORMAL_MEMORY_MANAGEMENT} )
add_executable(shared_memory_loop parallel/shared_memory/shared_memory_loop.cu ${COMMON} ${NORMAL_MEMORY_MANAGEMENT} )
add_executable(hybrid_memory parallel/hybrid_memory/hybrid_memory.cu ${COMMON} ${NORMAL_MEMORY_MANAGEMENT} )
add_executable(hybrid_memory_loop parallel/hybrid_memory/hybrid_memory_loop.cu ${COMMON} ${NORMAL_MEMORY_MANAGEMENT} )
add_executable(global_active_cells parallel/active_cells/global/global_active_cells.cu ${COMMON} ${NORMAL_MEMORY_MANAGEMENT} )
add_executable(global_active_cells_loop parallel/active_cells/global/global_active_cells_loop.cu ${COMMON} ${NORMAL_MEMORY_MANAGEMENT} )

# UNIFIED MEMORY MANAGEMENT
add_executable(unified_memory_loop parallel/unified_memory/unified_memory_loop.cu ${COMMON} ${UNIFIED_MEMORY_MANAGEMENT} )
add_executable(unified_memory parallel/unified_memory/unified_memory.cu ${COMMON} ${UNIFIED_MEMORY_MANAGEMENT} )
add_executable(active_cells parallel/active_cells/unified/unified_active_cells.cu ${COMMON} ${UNIFIED_MEMORY_MANAGEMENT} )
add_executable(active_cells_loop parallel/active_cells/unified/unified_active_cells_loop.cu ${COMMON} ${UNIFIED_MEMORY_MANAGEMENT} )
add_executable(river river/river_unified.cu ${COMMON} ${UNIFIED_MEMORY_MANAGEMENT} )
