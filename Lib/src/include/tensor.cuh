#include <iostream>

class Tensor
{
private:
    int rows;
    int cols;
    float *data;

public:
    Tensor(int rows, int cols);
    ~Tensor();
    // Copy constructor
    Tensor(const Tensor &other);
    // Assignment operator
    Tensor &operator=(const Tensor &other);
};
