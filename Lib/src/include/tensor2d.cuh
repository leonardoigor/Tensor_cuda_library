#ifndef TENSOR2d_CUR
#define TENSOR2d_CUR
#define BLOCK_SIZE 256

class Tensor2d
{
private:
    double *data;

    int cols;
    int rows;

public:
    Tensor2d(int cols, int rows);
    ~Tensor2d();
    static Tensor2d *Random(int min, int max, int cols, int rows);
    void print();
    Tensor2d *mul(double a);
    Tensor2d *setValue(double a);
};

#endif /*TENSOR2d_CUR*/