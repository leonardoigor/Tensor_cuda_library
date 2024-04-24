#include "include/tensor.cuh"
#include "tensor.cuh"
// #include "tensor.cuh"
// #include "tensor.cuh"
// #include "tensor.cuh"

Tensor::Tensor(int rows, int cols) : rows(rows), cols(cols)
{
    cudaMalloc(&data, rows * cols * sizeof(float));
}

Tensor::~Tensor()
{

    cudaFree(data);
}
// Copy constructor
Tensor::Tensor(const Tensor &other) : rows(other.rows), cols(other.cols)
{
    cudaMalloc(&data, rows * cols * sizeof(float));
    cudaMemcpy(data, other.data, rows * cols * sizeof(float), cudaMemcpyDeviceToDevice);
}

__host__ Tensor &Tensor::operator=(const Tensor &other)
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
__host__ int Tensor::getRows() const
{
    return rows;
}
__host__ int Tensor::getCols() const
{
    return cols;
}

__host__ void Tensor::setValues(float *hostData)
{
    cudaMemcpy(data, hostData, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void Tensor::print() const
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

__host__ void Tensor::toCPU(float *hostData) const
{
    cudaMemcpy(hostData, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
}

__device__ float &Tensor::operator()(int i, int j) const
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

__host__ void Tensor::Mul(Tensor *b, Tensor *c)
{
    printf("Test\n");
    auto cols = this->cols;
    auto rows = this->rows;
    int blockSize = 256;
    int numBlocks = (rows * cols + blockSize - 1) / blockSize;
    // c->print();

    MULGLOAL<<<numBlocks, blockSize>>>(this->data, b->data, c->data, cols, rows);
    cudaDeviceSynchronize();
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
}
// __device__ Tensor Tensor::operator+(const Tensor &other) const
// {
//     Tensor result(rows, cols);
//     for (int i = 0; i < rows; ++i)
//     {
//         for (int j = 0; j < cols; ++j)
//         {
//             result(i, j) = (*this)(i, j) + other(i, j);
//         }
//     }
//     return result;
// }

// __device__ Tensor Tensor::operator-(const Tensor &other) const
// {
//     Tensor result(*this);
//     for (int i = 0; i < rows; ++i)
//     {
//         for (int j = 0; j < cols; ++j)
//         {
//             result(i, j) = (*this)(i, j) - other(i, j);
//         }
//     }
//     return result;
// }

Tensor Tensor::operator*(Tensor &other)
{
    Tensor result(rows, other.cols);
    Tensor *b = &other;
    Tensor *result2 = &result;
    Mul(b, result2);
    return result;
}
