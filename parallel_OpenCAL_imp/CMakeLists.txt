cmake_minimum_required(VERSION 3.14)
project(cal_saturedFlow_oneWell)

set(CMAKE_CXX_STANDARD 14)

add_executable(cal_saturedFlow_oneWell cal_saturedFlow_oneWell.c)

target_link_libraries(cal_saturedFlow_oneWell opencal)
