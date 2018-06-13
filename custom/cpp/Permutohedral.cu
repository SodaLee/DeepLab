#define EIGEN_USE_GPU
#include "Permutohedral.hpp"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

#define CHECK_ERROR(error, line) {\
    cudaError_t err = error;\
    if(err != cudaSuccess){\
        fprintf(stderr, "error %d: %s\n", line, cudaGetErrorString(err));\
        exit(1);\
    }\
    }

template<typename T>
__global__ void splatting(
    const T *unary, int d,
    T *values, int np, const int *offset, const T *barycentric, int kernel_d
)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(; p < np; p += stride)
    {
        for(int j = 0; j < kernel_d+1; j++)
        {
            int o = offset[p * (kernel_d+1) + j] + 1;
            T w = barycentric[p * (kernel_d+1) + j];
            for(int k = 0; k < d; k++)
            {
                values[o * d + k] += w * unary[p * d + k];
            }
        }
    }
}

template<typename T>
__global__ void bluring(
    T *values, T *newval, int d, int kernel_d, int *neighbours, int M, bool reverse
)
{
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if(!reverse)
    {
        for(int j = 0; j < kernel_d+1; j++)
        {
            for(int i = i0; i < M; i += stride)
            {
                T *oldv = &values[(i+1) * d];
                T *newv = &newval[(i+1) * d];
                int *n = &neighbours[(j * M + i) * 2];
                int n1 = n[0] + 1;
                int n2 = n[1] + 1;
                T *n1v = &values[n1 * d];
                T *n2v = &values[n2 * d];
                for(int k = 0; k < d; k++)
                    newv[k] = oldv[k] + 0.5 * (n1v[k] + n2v[k]);
            }
            __syncthreads();
            T *t = newval;
            newval = values;
            values = t;
        }
    }
    else
    {
        for(int j = kernel_d; j >= 0; j--)
        {
            for(int i = i0; i < M; i += stride)
            {
                T *oldv = &values[(i+1) * d];
                T *newv = &newval[(i+1) * d];
                int *n = &neighbours[(j * M + i) * 2];
                int n1 = n[0] + 1;
                int n2 = n[1] + 1;
                T *n1v = &values[n1 * d];
                T *n2v = &values[n2 * d];
                for(int k = 0; k < d; k++)
                    newv[k] = oldv[k] + 0.5 * (n1v[k] + n2v[k]);
            }
            __syncthreads();
            T *t = newval;
            newval = values;
            values = t;
        }
    }
}

template<typename T>
__global__ void slicing(
    T *output, int d, bool add, T weight,
    const T *values, int np, const int *offset, const T *barycentric, int kernel_d
)
{
    T alpha = 1.0 / (1 + pow(2, -kernel_d));
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(; p < np; p += stride)
    {
        if(!add)
        {
            for(int k = 0; k < d; k++)
                output[p * d + k] = 0;
        }
        for(int j = 0; j < kernel_d+1; j++)
        {
            int o = offset[p * (kernel_d+1) + j] + 1;
            T w = barycentric[p * (kernel_d+1) + j];
            for(int k = 0; k < d; k++)
                output[p * d + k] += weight * w * values[o * d + k] * alpha;
        }
    }
}

template<typename T>
__host__ void Permutohedral<T>::compute(Tensor &output_tensor, const Tensor& unary_tensor, bool add, T weight, bool reverse, const GPUDevice& device)
{
    int batch_size = unary_tensor.dim_size(0),
        height = unary_tensor.dim_size(1),
        width = unary_tensor.dim_size(2),
        d = unary_tensor.dim_size(3);
    int np = height * width;
    const T *unary = unary_tensor.flat<T>().data();
    T *output = output_tensor.flat<T>().data();

    int *offset_kernel;
    T *barycentric_kernel;

    cudaMalloc(&offset_kernel, np * (kernel_d+1) * sizeof(int));
    cudaMalloc(&barycentric_kernel, np * (kernel_d+1) * sizeof(T));

    for(int b = 0; b < batch_size; b++)
    {
        int kb = b;
        if(kernel_batch == 1)
            kb = 0;
        int M = neighbours[kb].size() / (kernel_d+1);

        T *values_kernel, *newval_kernel;
        int *neighbours_kernel;
        const T *unary_kernel = &unary[b * np * d];
        T *output_kernel = &output[b * np * d];

        cudaMalloc(&values_kernel, (M+2) * d * sizeof(T));
        cudaMalloc(&newval_kernel, (M+2) * d * sizeof(T));
        cudaMalloc(&neighbours_kernel, M * (kernel_d+1) * sizeof(int) * 2);

        cudaMemset(values_kernel, 0, (M+2) * d * sizeof(T));
        cudaMemcpy(neighbours_kernel, neighbours[kb].data(), M * (kernel_d+1) * sizeof(int) * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(offset_kernel, &offset_[kb * np * (kernel_d+1)], np * (kernel_d+1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(barycentric_kernel, &barycentric_[kb * np * (kernel_d+1)], np * (kernel_d+1) * sizeof(T), cudaMemcpyHostToDevice);
        CHECK_ERROR(cudaGetLastError(), __LINE__);

        const int block_count = 512, thread_per_block = 32;
        splatting<<<block_count, thread_per_block>>>(
            unary_kernel, d, values_kernel, np,
            offset_kernel, barycentric_kernel, kernel_d);
        CHECK_ERROR(cudaGetLastError(), __LINE__);

        bluring<<<block_count, thread_per_block>>>(
            values_kernel, newval_kernel, d, kernel_d, neighbours_kernel, M, reverse);
        if(kernel_d % 2 == 0)
            std::swap(values_kernel, newval_kernel);
        CHECK_ERROR(cudaGetLastError(), __LINE__);
        
        slicing<<<block_count, thread_per_block>>>(
            output_kernel, d, add, weight, values_kernel, np,
            offset_kernel, barycentric_kernel, kernel_d);
        CHECK_ERROR(cudaGetLastError(), __LINE__);

        cudaFree(values_kernel);
        cudaFree(newval_kernel);
        cudaFree(neighbours_kernel);
    }
    cudaFree(offset_kernel);
    cudaFree(barycentric_kernel);
}

template class Permutohedral<float>;
template class Permutohedral<double>;

