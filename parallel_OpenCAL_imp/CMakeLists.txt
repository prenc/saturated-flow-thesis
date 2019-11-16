cmake_minimum_required(VERSION 3.15)
project(cal_saturedFlow_oneWell)

set(CMAKE_CXX_STANDARD 14)

add_executable(cal_saturedFlow_oneWell
        source/cal_saturedFlow_oneWell.c)

target_link_libraries(cal_saturedFlow_oneWell opencal)
