cmake_minimum_required(VERSION 2.8)
project(TensorTest)

find_package(CUDA REQUIRED)
include_directories("D:/Devs/TensorLib/Lib/src/include")

set(CUDA_NVCC_FLAGS "-arch=sm_80")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

include_directories(
    include
)

cuda_add_executable(${PROJECT_NAME}_APP
    main.cpp)
get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." REALPATH)

# Link your executable with the precompiled library
target_link_libraries(${PROJECT_NAME}_APP ${PARENT_DIR}/build/Lib/Debug/Tensor.lib ${PARENT_DIR}/build/Lib/Debug/Tensor_CUDA.lib)