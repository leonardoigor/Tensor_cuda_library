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

        // // Generate a random value within the specified range
        printf("[(%d,%d) (%d,%d)] ", row, rows, col, cols);
        float random_value = min + (max - min) * curand_uniform(&state);
        // results[row][col] = random_value;
        printf(" r= (%f,%f)", random_value, results[col][row]);
    }
}
// Function to generate random number between min and max
__device__ double generateRandomNumber(double min, double max, curandState_t *state)
{
    // Generate random number within [min, max] range
    return min + (max - min) * curand_uniform_double(state);
}
__global__ void CreateRandomListMinMax(int min, int max, int cols, int rows, double *data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    curandState_t state;
    curand_init(clock64(), tid, 0, &state);
    for (int i = tid; i < rows * cols; i += stride)
    {
        auto value = generateRandomNumber(min, max, &state);
        data[i] = value;
    }
}
// Kernel to multiply each element of data with a double value
__global__ void MultiplyData(double multiplier, int cols, int rows, double *data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < rows * cols; i += stride)
    {
        data[i] *= multiplier;
    }
}
__global__ void SetData(double val, int cols, int rows, double *data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < rows * cols; i += stride)
    {
        data[i] = val;
    }
}
