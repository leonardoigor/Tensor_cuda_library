#ifndef tensor2d_kerne_CUR
#define tensor2d_kerne_CUR
#include <curand_kernel.h>

__global__ void RandomInRangeKernel(int min, int max, int cols, int rows, float **results);

// Function to generate random number between min and max
__device__ double generateRandomNumber(double min, double max, curandState_t *state);
__global__ void CreateRandomListMinMax(int min, int max, int cols, int rows, double *data);
// Kernel to multiply each element of data with a double value
__global__ void MultiplyData(double multiplier, int cols, int rows, double *data);

__global__ void SetData(double val, int cols, int rows, double *data);
#endif