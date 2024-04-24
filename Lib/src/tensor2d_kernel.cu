#include "tensor2d_kernel.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void RandomInRangeKernel(int min, int max, int cols, int rows, float **results)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        curandState state;
        curand_init(0, row * cols + col, 0, &state); // Initialize curand state

        // Generate a random value within the specified range
        float random_value = min + (max - min) * curand_uniform(&state);
        results[row][col] = random_value;
    }
}
