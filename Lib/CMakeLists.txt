cmake_minimum_required(VERSION 2.8)
project(Tensor)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "-arch=sm_80")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

include_directories(
    src/include
)

# Collect C++ source files
file(GLOB CPP_FILES src/*.cpp)

# Collect CUDA source files
file(GLOB CUDA_FILES src/*.cu)

foreach(file ${CPP_FILES})
    message(STATUS "Building : [${file}]")
endforeach()

foreach(file ${CUDA_FILES})
    message(STATUS "Building : [${file}]")
endforeach()

# Compile CUDA source files
cuda_add_library(${PROJECT_NAME}_CUDA
    ${CUDA_FILES}
)

# Compile C++ source files
add_library(${PROJECT_NAME}
    ${CPP_FILES}
)

# Include directories for the CUDA library
target_include_directories(${PROJECT_NAME}_CUDA
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
    $<INSTALL_INTERFACE:src/include>
)

# Include directories for the C++ library
target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
    $<INSTALL_INTERFACE:src/include>
)

# Link CUDA library with C++ library
target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${PROJECT_NAME}_CUDA
)