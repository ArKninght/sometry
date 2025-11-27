#include <torch/extension.h>
#include <torch/torch.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// cuda declaration
void Fista_fproj(torch::Tensor sinogram,torch::Tensor volume, torch::Tensor projVector, torch::Tensor volumeSize,
    torch::Tensor  projSize, torch::Tensor volinfo,int device);

void Fista_resprocess(torch::Tensor out, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
    const float volbiasz, const float dSampleInterval, const float dSliceInterval, const long device);

void Fista_bproj(torch::Tensor volume_P, torch::Tensor VolumeBackProj, torch::Tensor residual, torch::Tensor projVector, torch::Tensor volumeSize,
    torch::Tensor  projSize, torch::Tensor volinfo,int device);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fproj", &Fista_fproj, "Fista fproj (CUDA)");
    m.def("bproj", &Fista_bproj,"Fista bproj (CUDA)");
    m.def("resprocess", &Fista_resprocess,"Fista resprocess (CUDA)");

}