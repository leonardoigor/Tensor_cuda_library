#include "include/Tensor.cuh"
#include "Tensor.cuh"
// #include "Tensor1d.cuh"
// #include "Tensor1d.cuh"
// #include "Tensor1d.cuh"

Tensor1d::Tensor1d(int rows, int cols) : rows(rows), cols(cols)
{
    cudaMalloc(&data, rows * cols * sizeof(float));
}

Tensor1d::~Tensor1d()
{

    cudaFree(data);
}
// Copy constructor
Tensor1d::Tensor1d(const Tensor1d &other) : rows(other.rows), cols(other.cols)
{
    cudaMalloc(&data, rows * cols * sizeof(float));
    cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
}

__host__ Tensor1d &Tensor1d::operator=(const Tensor1d &other)
{
    if (this != &other)
    {
        cudaFree(data);
        rows = other.rows;
        cols = other.cols;
        cudaMalloc(&data, rows * cols * sizeof(float));
        cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return *this;
}
__host__ int Tensor1d::getRows() const
{
    return rows;
}
__host__ int Tensor1d::getCols() const
{
    return cols;
}

__host__ void Tensor1d::setValues(float *hostData)
{
    cudaMemcpy(data, hostData, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void Tensor1d::print() const
{
    float *hostData = new float[rows * cols];
    cudaMemcpy(hostData, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << hostData[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] hostData;
}

__host__ void Tensor1d::toCPU(float *hostData) const
{
    cudaMemcpy(hostData, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
}

__device__ float &Tensor1d::operator()(int i, int j) const
{
    return data[i * cols + j];
}

__global__ void MULGLOAL(float *a, float *b, float *c, int cols, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; ++j)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid < rows * cols)
            {
                c[tid] = a[tid] * b[tid];
                // printf("\n----%d,%d--(%f)-\n", j, i, c[tid]);
            }
        }
    }
    // printf("\n----%d,%d---\n", cols, rows);
}
__global__ void SUMGLOAL(float *a, float *b, float *c, int cols, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; ++j)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid < rows * cols)
            {
                // printf("d %f\n", c[tid]);
                c[tid] = a[tid] + b[tid];
                // printf("a %f\n", c[tid]);
            }
        }
    }
}

__host__ void Tensor1d::Mul(Tensor1d *b, Tensor1d *c)
{
    // printf("Test\n");
    auto cols = this->cols;
    auto rows = this->rows;
    int blockSize = 256;
    int numBlocks = (rows * cols + blockSize - 1) / blockSize;
    // c->print();

    MULGLOAL<<<numBlocks, blockSize>>>(this->data, b->data, c->data, cols, rows);
    cudaDeviceSynchronize();
}
__host__ void Tensor1d::Sum(Tensor1d *b, Tensor1d *c)
{
    // printf("Test\n");
    auto cols = this->cols;
    auto rows = this->rows;
    int blockSize = 256;
    int numBlocks = (rows * cols + blockSize - 1) / blockSize;
    // c->print();

    SUMGLOAL<<<numBlocks, blockSize>>>(this->data, b->data, c->data, cols, rows);
    cudaDeviceSynchronize();
}
// c->print();
// cudaError_t cudaError = cudaGetLastError();
// if (cudaError != cudaSuccess)
// {
//     std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
// }
// else
// {
//     std::cout << "No CUDA error detected." << std::endl;
// }
Tensor1d Tensor1d::operator+(Tensor1d &other)
{
    Tensor1d result(rows, cols);
    Tensor1d *b = &other;
    Tensor1d *result2 = &result;
    Sum(b, result2);
    return result;
}

// __device__ Tensor1d Tensor1d::operator-(const Tensor1d &other) const
// {
//     Tensor1d result(*this);
//     for (int i = 0; i < rows; ++i)
//     {
//         for (int j = 0; j < cols; ++j)
//         {
//             result(i, j) = (*this)(i, j) - other(i, j);
//         }
//     }
//     return result;
// }

Tensor1d Tensor1d::operator*(Tensor1d &other)
{
    Tensor1d result(rows, other.cols);
    Tensor1d *b = &other;
    Tensor1d *result2 = &result;
    Mul(b, result2);
    return result;
}
