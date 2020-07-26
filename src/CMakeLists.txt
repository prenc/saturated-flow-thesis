cmake_minimum_required(VERSION 3.16)

project(saturated_flow_thesis)
enable_language(CUDA)

add_executable(global_memory parallel/global_memory/global_memory.cu)
add_executable(global_memory_loop parallel/global_memory/global_memory_loop.cu)
add_executable(shared_memory parallel/shared_memory/shared_memory.cu)
add_executable(shared_memory_loop parallel/shared_memory/shared_memory_loop.cu)
add_executable(hybrid_memory parallel/hybrid_memory/hybrid_memory.cu)
add_executable(hybrid_memory_loop parallel/hybrid_memory/hybrid_memory_loop.cu)
add_executable(unified_memory_loop parallel/unified_memory/unified_memory_loop.cu)
add_executable(unified_memory parallel/unified_memory/unified_memory.cu)
add_executable(active_cells parallel/active_cells/unified/unified_active_cells.cu)
add_executable(active_cells_loop parallel/active_cells/unified/unified_active_cells_loop.cu)
add_executable(river river/river_unified.cu)