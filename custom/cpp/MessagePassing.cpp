#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include <type_traits>

#include "Permutohedral.hpp"

using namespace tensorflow;
using namespace std;

template<typename T>
void get_bilateral_kernel(T *const output_kernel, const T *raw, int batch, int height, int width, int channel,
    const T theta_alpha, const T theta_beta)
{
    int stride = channel + 2;
    int np = height * width;
    for(int i = 0; i < batch; i++)
    {
        T *b = &output_kernel[i * np * stride];
        for(int p = 0; p < np; p++)
        {
            b[stride * p] = static_cast<T>(p / width) / (height - 1) / theta_alpha;
            b[stride * p + 1] = static_cast<T>(p % width) / (width - 1) / theta_alpha;
            for(int c = 0; c < channel; c++)
            {
                b[stride * p + c + 2] = raw[(i * np + p) * stride + c] / theta_beta;
            }
        }
    }
}

template<typename T>
void get_spatial_kernel(T *const output_kernel, int height, int width,
    const T theta_gamma)
{
    int np = height * width;
    for(int i = 0; i < np; i++)
    {
        output_kernel[i * 2] = static_cast<T>(i / width) / (height - 1) / theta_gamma;
        output_kernel[i * 2 + 1] = static_cast<T>(i % width) / (width - 1) / theta_gamma;
    }
}

template<typename T>
extern void get_tensor_gpu(T *&, const T*, int);

REGISTER_OP("MessagePassing")
    .Attr("T: {float, double}")
    .Attr("reverse: bool = false")
    .Input("unary: T")
    .Input("raw: T")
    .Input("kernels: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        ::tensorflow::shape_inference::ShapeHandle output;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &output));
        c->set_output(0, output);
        return Status::OK();
    });

template<typename Device, typename T>
class MessagePassingOp: public OpKernel
{
    bool reverse;
public:
    explicit MessagePassingOp(OpKernelConstruction* context): OpKernel(context)
    {
        OP_REQUIRES_OK(context,
            context->GetAttr("reverse", &reverse));
    }
    void Compute(OpKernelContext* context) override
    {
        const Tensor& unary_tensor = context->input(0);
        const Tensor& raw_tensor = context->input(1);
        const Tensor& kernels_tensor = context->input(2);

        CHECK_EQ(unary_tensor.dims(), 4);
        CHECK_EQ(raw_tensor.dims(), 4);
        CHECK_EQ(kernels_tensor.dim_size(0), 5);
        for(int i = 0; i < 3; i++)
            CHECK_EQ(unary_tensor.dim_size(i), raw_tensor.dim_size(i));
        
        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, unary_tensor.shape(), &output_tensor));

        int batch_size = unary_tensor.dim_size(0),
            height = unary_tensor.dim_size(1),
            width = unary_tensor.dim_size(2),
            nclass = unary_tensor.dim_size(3),
            nchann = raw_tensor.dim_size(3);

        T *kernels, *raw;
        if(is_same<Device, GPUDevice>::value)
        {
            get_tensor_gpu(kernels, kernels_tensor.flat<T>().data(), 5);
            get_tensor_gpu(raw, raw_tensor.flat<T>().data(), raw_tensor.NumElements());
        }
        else
        {
            kernels = new T[5];
            memcpy(kernels, kernels_tensor.flat<T>().data(), 5 * sizeof(T));
            raw = const_cast<T*>(raw_tensor.flat<T>().data());
        }

        Permutohedral<T> p;
        T *kernel = new T[batch_size * height * width * (nchann + 2)];
        
        get_bilateral_kernel(kernel, raw, batch_size, height, width, nchann, kernels[0], kernels[1]);
        p.init(kernel, nchann + 2, batch_size, height * width);
        p.compute(*output_tensor, unary_tensor, false, kernels[3], reverse, context->eigen_device<Device>());

        get_spatial_kernel(kernel, height, width, kernels[2]);
        p.clear();
        p.init(kernel, 2, 1, height * width);
        p.compute(*output_tensor, unary_tensor, true, kernels[4], reverse, context->eigen_device<Device>());
        delete[] kernel;
        delete[] kernels;
        if(is_same<Device, GPUDevice>::value)
        {
            delete[] raw;
        }
    }
};

REGISTER_KERNEL_BUILDER(
    Name("MessagePassing")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    MessagePassingOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("MessagePassing")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    MessagePassingOp<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(
    Name("MessagePassing")
    .Device(DEVICE_GPU)
    .TypeConstraint<float>("T"),
    MessagePassingOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("MessagePassing")
    .Device(DEVICE_GPU)
    .TypeConstraint<double>("T"),
    MessagePassingOp<GPUDevice, double>);
