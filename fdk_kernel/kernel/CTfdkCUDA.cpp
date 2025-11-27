#include <torch/extension.h>
#include <torch/torch.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// cuda declaration
void forward(torch::Tensor volume, torch::Tensor sinogram ,torch::Tensor projectVector, int device);
void backward(torch::Tensor volume, torch::Tensor sinogram ,torch::Tensor projectVector, int device);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fproj", &forward, "fproj (CUDA)");
    m.def("bproj", &backward,"bproj (CUDA)");
}