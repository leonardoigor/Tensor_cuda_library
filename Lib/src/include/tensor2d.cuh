#ifndef TENSOR2d_CUR
#define TENSOR2d_CUR

class Tensor2d
{
private:
    float **data;

    int cols;
    int rows;

public:
    Tensor2d(int cols, int rows);
    ~Tensor2d();
    static Tensor2d Random(int min, int max, int cols, int rows);
};

#endif /*TENSOR2d_CUR*/