#include "tensorflow/core/framework/tensor.h"
using namespace tensorflow;

template<class T>
__host__ void get_tensor_gpu(T *&output, const T* d, int size)
{
    output = new T[size];
    cudaMemcpy(output, d, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template void get_tensor_gpu(float *&, const float*, int);
template void get_tensor_gpu(double *&, const double*, int);

