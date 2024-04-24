cmake_minimum_required(VERSION 2.8)
project(Main)
find_package(CUDA REQUIRED)
add_subdirectory(Lib)

# execute_process(
# COMMAND ${CMAKE_COMMAND} -E sleep 2
# )
add_subdirectory(test)
