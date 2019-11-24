cmake_minimum_required(VERSION 3.14)
project(cal_saturatedFlow_oneWell)

set(CMAKE_CXX_STANDARD 14)

add_executable(cal_saturatedFlow_oneWell cal_saturatedFlow_oneWell.c)

target_link_libraries(cal_saturatedFlow_oneWell opencal)
