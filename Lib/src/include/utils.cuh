#define CUDA_CHECK(call)                                                              \
    do                                                                                \
    {                                                                                 \
        cudaError_t error = call;                                                     \
        if (error != cudaSuccess)                                                     \
        {                                                                             \
            printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n",                       \
                   __FILE__, __LINE__, (int)error, cudaGetErrorString(error), #call); \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)
