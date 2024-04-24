#ifndef TENSOR_CUR
#define TENSOR_CUR
#ifdef __CUDACC__ // Check if compiling with NVCC (CUDA compiler)
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <iostream>

class Tensor
{
private:
    int rows;
    int cols;

public:
    float *data;
    // Constructor
    Tensor(int rows, int cols);

    // Destructor
    ~Tensor();

    // Copy constructor
    Tensor(const Tensor &other);

    // Assignment operator
    Tensor &operator=(const Tensor &other);

    // Getters
    int getRows() const;
    int getCols() const;

    // Function to set tensor values
    void setValues(float *hostData);

    // Function to print tensor values
    void print() const;

    // Function to copy tensor data from GPU to CPU
    void toCPU(float *hostData) const;

    // Overloaded operators for element-wise operations

    float &operator()(int i, int j) const;

    void Tensor::Mul(Tensor *b, Tensor *c);
    // __device__ Tensor operator+(const Tensor &other) const;

    // __device__ Tensor operator-(const Tensor &other) const;

    // // Matrix multiplication
    Tensor operator*(Tensor &other);
};

#endif /*TENSOR_CUR*/