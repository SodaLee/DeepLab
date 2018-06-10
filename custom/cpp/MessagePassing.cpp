#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "Permutohedral.hpp"

using namespace tensorflow;

template<typename T>
void get_bilateral_kernel(T *const output_kernel, const Tensor& raw_tensor,
    const T theta_alpha, const T theta_beta)
{
    CHECK_EQ(raw_tensor.dims(), 4) << "Compute Kernel: Only accept 4D tensor";
    const int batch = raw_tensor.dim_size(0),
              height = raw_tensor.dim_size(1),
              width = raw_tensor.dim_size(2),
              channel = raw_tensor.dim_size(3);
    const int np = height * width;
    const int stride = channel + 2;
    auto raw = raw_tensor.tensor<T, 4>();
    for(int i = 0; i < batch; i++)
    {
        T *b = &output_kernel[i * np * stride];
        for(int p = 0; p < np; p++)
        {
            b[stride * p] = static_cast<T>(p / width) / (height - 1) / theta_alpha;
            b[stride * p + 1] = static_cast<T>(p % width) / (width - 1) / theta_alpha;
            for(int c = 0; c < channel; c++)
            {
                b[stride * p + c + 2] = raw(i, p / width, p % width, c) / theta_beta;
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

template<typename T>
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

        auto kernels = kernels_tensor.flat<T>();

        Permutohedral<T> p;
        T *kernel = new T[batch_size * height * width * (nchann + 2)];
        
        get_bilateral_kernel(kernel, raw_tensor, kernels(0), kernels(1));
        p.init(kernel, nchann + 2, batch_size, height * width);
        p.compute(*output_tensor, unary_tensor, false, kernels(3), reverse);

        get_spatial_kernel(kernel, height, width, kernels(2));
        p.clear();
        p.init(kernel, 2, 1, height * width);
        p.compute(*output_tensor, unary_tensor, true, kernels(4), reverse);
        delete[] kernel;
    }
};

REGISTER_KERNEL_BUILDER(
    Name("MessagePassing")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    MessagePassingOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("MessagePassing")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    MessagePassingOp<double>);
