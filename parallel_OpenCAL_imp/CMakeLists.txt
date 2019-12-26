cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(cal_saturatedFlow_oneWell C)
find_package(OpenCAL 1.0 COMPONENTS OpenCAL REQUIRED)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${OPENCAL_INCLUDE_DIR}
                    )
FIND_PACKAGE(OpenMP REQUIRED)

IF(OPENMP_FOUND)
		SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()

file(GLOB SRC
    "include/*.h"
    "*.c"
)

add_executable(cal_saturatedFlow_oneWell ${SRC})
target_link_libraries(cal_saturatedFlow_oneWell ${OPENCAL_LIBRARIES})
