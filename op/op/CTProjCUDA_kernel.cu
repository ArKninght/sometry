#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include "helper_math.h"
#define BLOCK_X 32
#define BLOCK_Y 32
#define PI 3.14159265359

namespace {
    __global__ void fproj(float* sinogram_device, float* projVector_device, int* volumeSize_device, int* projSize_device,float* volinfo_device, int anglenums,
        cudaTextureObject_t my_tex, int angle) {
        uint3 detectorIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
        if (detectorIdx.x > projSize_device[1] || detectorIdx.y > projSize_device[0]) {
            return;
        }
        float2 detectorCenter = make_float2(projSize_device[1] / -2.0, projSize_device[0] / -2.0);
        float3 volumeCenter = make_float3(volumeSize_device[2] / -2.0, volumeSize_device[1] / -2.0, volumeSize_device[0] / -2.0);
        float dSampleInterval = volinfo_device[1];
        float dSliceInterval = volinfo_device[2];
        float volbiasz = volinfo_device[0];
        // 获取探测器坐标(包含负值的实际坐标)
        float detectorX = detectorIdx.x + detectorCenter.x;
        float detectorY = detectorIdx.y + detectorCenter.y;
        //for (int angle = 0; angle < anglenums; angle++) {
        int id_proj = angle * projSize_device[0] * projSize_device[1] + detectorIdx.y * projSize_device[1] + detectorIdx.x;
        float3 sourcePosition = make_float3(projVector_device[angle * 12], projVector_device[angle * 12 + 1], projVector_device[angle * 12 + 2]);
        float3 detectorPosition = make_float3(projVector_device[angle * 12 + 3], projVector_device[angle * 12 + 4], projVector_device[angle * 12 + 5]);
        float3 u = make_float3(projVector_device[angle * 12 + 6], projVector_device[angle * 12 + 7], projVector_device[angle * 12 + 8]);
        float3 v = make_float3(projVector_device[angle * 12 + 9], projVector_device[angle * 12 + 10], projVector_device[angle * 12 + 11]);

        float3 detectorPixel = detectorPosition + (detectorX+0.5) * u + (detectorY+0.5) * v;
        float3 rayVector = normalize(detectorPixel - sourcePosition);

        float pixel = 0.0f;
        float alpha0, alpha1;
        float rayVectorDomainDim = fmax(fabs(rayVector.x), fmax(fabs(rayVector.z), fabs(rayVector.y)));
        if (fabs(rayVector.x) == rayVectorDomainDim) {
            float volume_min_edge_point = volumeCenter.x * dSampleInterval;
            float volume_max_edge_point = (volumeSize_device[2] + volumeCenter.x) * dSampleInterval;
            alpha0 = (volume_min_edge_point - sourcePosition.x) / rayVector.x;
            alpha1 = (volume_max_edge_point - sourcePosition.x) / rayVector.x;
        }
        else if (fabs(rayVector.y) == rayVectorDomainDim) {
            float volume_min_edge_point = volumeCenter.y * dSampleInterval;
            float volume_max_edge_point = (volumeSize_device[1] + volumeCenter.y) * dSampleInterval;
            alpha0 = (volume_min_edge_point - sourcePosition.y) / rayVector.y;
            alpha1 = (volume_max_edge_point - sourcePosition.y) / rayVector.y;
        }
        else {
            float volume_min_edge_point = volumeCenter.z * dSliceInterval + volbiasz;
            float volume_max_edge_point = (volumeSize_device[0] + volumeCenter.z) * dSliceInterval + volbiasz;
            alpha0 = (volume_min_edge_point - sourcePosition.z) / rayVector.z;
            alpha1 = (volume_max_edge_point - sourcePosition.z) / rayVector.z;
        }

        float min_alpha = fmin(alpha0, alpha1) - 3;
        float max_alpha = fmax(alpha0, alpha1) + 3;
        float px, py, pz;
        float step_size = 1;

        while (min_alpha < max_alpha)
        {
            px = sourcePosition.x + min_alpha * rayVector.x;
            py = sourcePosition.y + min_alpha * rayVector.y;
            pz = sourcePosition.z + min_alpha * rayVector.z - volbiasz;
            px /= dSampleInterval;
            py /= dSampleInterval;
            pz /= dSliceInterval;
            px -= volumeCenter.x;
            py -= volumeCenter.y;
            pz -= volumeCenter.z;
            pixel += tex3D<float>(my_tex, px + 0.5f, py + 0.5f, pz + 0.5f);
            min_alpha += step_size;
        }
        pixel *= step_size;
        sinogram_device[id_proj] = (pixel) ;
        return;
    }

    __global__ void resproKernel(float* res, const uint3 volumeSize, const float3 volumeCenter, const uint2 detectorSize, const float2 detectorCenter, const float* projectVector, const uint index,
        const float volbiasz, const float dSampleInterval, const float dSliceInterval) {
        uint3 detectorIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
        if (detectorIdx.x >= detectorSize.x || detectorIdx.y >= detectorSize.y) {
            return;
        }

        float detectorX = detectorIdx.x + detectorCenter.x;
        float detectorY = detectorIdx.y + detectorCenter.y;

        float3 sourcePosition = make_float3(projectVector[index * 12], projectVector[index * 12 + 1], projectVector[index * 12 + 2]);
        float3 detectorPosition = make_float3(projectVector[index * 12 + 3], projectVector[index * 12 + 4], projectVector[index * 12 + 5]);
        float3 u = make_float3(projectVector[index * 12 + 6], projectVector[index * 12 + 7], projectVector[index * 12 + 8]);
        float3 v = make_float3(projectVector[index * 12 + 9], projectVector[index * 12 + 10], projectVector[index * 12 + 11]);

        float3 detectorPixel = detectorPosition + (detectorX) * u + (detectorY) * v;
        float3 rayVector = normalize(detectorPixel - sourcePosition);

        unsigned resIdx = index * detectorSize.x * detectorSize.y + detectorIdx.y * detectorSize.x + detectorIdx.x;
        float pixel = res[resIdx];
        float alpha0, alpha1;
        float rayVectorDomainDim = fmax(fabs(rayVector.x), fmax(fabs(rayVector.z), fabs(rayVector.y)));
        if (fabs(rayVector.x) == rayVectorDomainDim) {
            float volume_min_edge_point = volumeCenter.x * dSampleInterval;
            float volume_max_edge_point = (volumeSize.x + volumeCenter.x) * dSampleInterval;
            alpha0 = (volume_min_edge_point - sourcePosition.x) / rayVector.x;
            alpha1 = (volume_max_edge_point - sourcePosition.x) / rayVector.x;
        }
        else if (fabs(rayVector.y) == rayVectorDomainDim) {
            float volume_min_edge_point = volumeCenter.y * dSampleInterval;
            float volume_max_edge_point = (volumeSize.y + volumeCenter.y) * dSampleInterval;
            alpha0 = (volume_min_edge_point - sourcePosition.y) / rayVector.y;
            alpha1 = (volume_max_edge_point - sourcePosition.y) / rayVector.y;
        }
        else {
            float volume_min_edge_point = volumeCenter.z * dSliceInterval + volbiasz;
            float volume_max_edge_point = (volumeSize.z + volumeCenter.z) * dSliceInterval + volbiasz;
            alpha0 = (volume_min_edge_point - sourcePosition.z) / rayVector.z;
            alpha1 = (volume_max_edge_point - sourcePosition.z) / rayVector.z;
        }
        float min_alpha = fmin(alpha0, alpha1) - 3;
        float max_alpha = fmax(alpha0, alpha1) + 3;
        float len = max_alpha - min_alpha;

        pixel /= len;
        res[resIdx] = pixel;

        return;
    }


    __global__ void bproj(float* VolumeBackProj_device, float* residual_device, int* volumeSize_device, int* projSize_device, float* projVector_device, float* volinfo_device, int anglenums, cudaTextureObject_t my_tex, int angle) {
        uint3 volumeIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
        if (volumeIdx.x > volumeSize_device[2] || volumeIdx.y > volumeSize_device[1]) {
            return;
        }
        float3 sourcePosition = make_float3(projVector_device[angle * 12], projVector_device[angle * 12 + 1], projVector_device[angle * 12 + 2]);
        float3 detectorPosition = make_float3(projVector_device[angle * 12 + 3], projVector_device[angle * 12 + 4], projVector_device[angle * 12 + 5]);
        float3 u = make_float3(projVector_device[angle * 12 + 6], projVector_device[angle * 12 + 7], projVector_device[angle * 12 + 8]);
        float3 v = make_float3(projVector_device[angle * 12 + 9], projVector_device[angle * 12 + 10], projVector_device[angle * 12 + 11]);
        float3 volumeCenter = make_float3(volumeSize_device[2] / -2.0f, volumeSize_device[1] / -2.0f, volumeSize_device[0] / -2.0f);
        float2 detectorCenter = make_float2(projSize_device[1] / -2.0f, projSize_device[0] / -2.0f);
        float dSampleInterval = volinfo_device[1];
        float dSliceInterval = volinfo_device[2];
        float volbiasz = volinfo_device[0];
        for (int z0 = 0; z0 < volumeSize_device[0]; z0++) {
            int volume_id = z0 * volumeSize_device[1] * volumeSize_device[2] + volumeIdx.y * volumeSize_device[2] + volumeIdx.x;
            float3 coordinates = make_float3((volumeCenter.x + volumeIdx.x) * dSampleInterval, (volumeCenter.y + volumeIdx.y) * dSampleInterval, (volumeCenter.z + z0) * dSliceInterval + volbiasz);
            float fScale = __fdividef(1.0f, det3(u, v, sourcePosition - coordinates));
            fScale = det3(u, v, sourcePosition - coordinates) == 0 ? 0 : fScale;
            float detectorX = fScale * det3(coordinates - sourcePosition, v, sourcePosition - detectorPosition) - detectorCenter.x;
            float detectorY = fScale * det3(u, coordinates - sourcePosition, sourcePosition - detectorPosition) - detectorCenter.y;
            VolumeBackProj_device[volume_id] += (tex3D<float>(my_tex, detectorX, detectorY, angle + 0.5f) / anglenums);

	    }

        return;
    }

}

void Fista_fproj(torch::Tensor sinogram, torch::Tensor volume,  torch::Tensor projVector,
 torch::Tensor volumeSize,torch::Tensor  projSize, torch::Tensor volinfo,int device) {

    int anglenums = sinogram.size(0);
    float* sinogram_device = sinogram.data<float>();
    torch::Device Device(torch::kCUDA, device);

    float* projVector_device = projVector.data<float>();
    int* volumeSize_device = volumeSize.data<int>();
    int* projSize_device = projSize.data<int>();
    float* volinfo_device = volinfo.data<float>();
    float* volume_device = volume.data<float>();
    int projWidth = sinogram.size(2);
    int projHeight = sinogram.size(1);
    //torch::Tensor residual = torch::zeros({anglenums,projHeight,projWidth}, torch::kFloat).to(Device);
    int volumeWidth = volume.size(2);
    int volumeHeight = volume.size(0);
    int volumeDepth = volume.size(1);
    //初始化纹理对象
    const cudaChannelFormatDesc channelDesc_vol = cudaCreateChannelDesc<float>();
    cudaArray* d_cube_array_vol;
    cudaMalloc3DArray(&d_cube_array_vol, &channelDesc_vol, make_cudaExtent(volumeWidth, volumeDepth, volumeHeight));
    cudaMemcpy3DParms copyParams_vol = { 0 };
    copyParams_vol.dstArray = d_cube_array_vol;
    copyParams_vol.extent = make_cudaExtent(volumeWidth, volumeDepth, volumeHeight);
    copyParams_vol.kind = cudaMemcpyDeviceToDevice;
    cudaTextureDesc     texDescr_vol;
    memset(&texDescr_vol, 0, sizeof(cudaTextureDesc));
    texDescr_vol.normalizedCoords = false;
    texDescr_vol.filterMode = cudaFilterModeLinear;
    texDescr_vol.addressMode[0] = cudaAddressModeBorder;
    texDescr_vol.addressMode[1] = cudaAddressModeBorder;
    texDescr_vol.addressMode[2] = cudaAddressModeBorder;
    texDescr_vol.readMode = cudaReadModeElementType;

    cudaResourceDesc    texRes_vol;
    memset(&texRes_vol, 0, sizeof(cudaResourceDesc));
    texRes_vol.resType = cudaResourceTypeArray;
    // 将volume数据绑定到纹理内存
    cudaTextureObject_t tex_vol;
    copyParams_vol.srcPtr = make_cudaPitchedPtr(volume_device, volumeWidth * sizeof(float), volumeWidth, volumeDepth);
    cudaMemcpy3D(&copyParams_vol);
    texRes_vol.res.array.array = d_cube_array_vol;
    cudaCreateTextureObject(&tex_vol, &texRes_vol, &texDescr_vol, NULL);
    // 获取线程块数和模块数

    const dim3 blockSize_proj = dim3(BLOCK_X, BLOCK_Y);
    const dim3 gridSize_proj = dim3(( projWidth + blockSize_proj.x - 1) / blockSize_proj.x, (projHeight + blockSize_proj.y - 1) / blockSize_proj.y);
    for (int angle = 0; angle < anglenums; angle++) {
        //("num:%d\n", angle);
        fproj << < gridSize_proj, blockSize_proj >> > (sinogram_device, projVector_device, volumeSize_device, projSize_device,
           volinfo_device, anglenums, tex_vol, angle);
    }
    cudaDestroyTextureObject(tex_vol);
    cudaFreeArray(d_cube_array_vol);
    //return residual;
    return;
}

void Fista_resprocess(torch::Tensor out, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
    const float volbiasz, const float dSampleInterval, const float dSliceInterval, const long device) {

    int angles = projectVector.size(0);
    float* outPtr = out.data<float>();

    uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
    float3 volumeCenter = make_float3(volumeSize) / -2.0;
    uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
    float2 detectorCenter = make_float2(detectorSize) / -2.0;
    float* outPtrPitch = outPtr;
    const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, 1);
    const dim3 gridSize = dim3(detectorSize.x / BLOCK_X + 1, detectorSize.y / BLOCK_Y + 1, 1);
    for (int angle = 0; angle < angles; angle++) {
        resproKernel << <gridSize, blockSize >> > (outPtrPitch, volumeSize, volumeCenter, detectorSize, detectorCenter, (float*)projectVector.data<float>(), angle,
            volbiasz, dSampleInterval, dSliceInterval);
    }

    return;
}

void Fista_bproj(torch::Tensor volume_P, torch::Tensor VolumeBackProj, torch::Tensor residual, torch::Tensor projVector, torch::Tensor volumeSize,
    torch::Tensor  projSize, torch::Tensor volinfo,int device) {

    int anglenums = residual.size(0);
    int projWidth = residual.size(2);
    int projDepth = residual.size(1);
    torch::Device Device(torch::kCUDA, device);

    float* residual_device = residual.data<float>();
    float* projVector_device = projVector.data<float>();
    int* volumeSize_device = volumeSize.data<int>();
    int* projSize_device = projSize.data<int>();
    float* volinfo_device = volinfo.data<float>();
    int volWidth = volume_P.size(2);
    int volDepth = volume_P.size(1);
    int volHeight = volume_P.size(0);
    //initialize proj Texture
    // 将projection数据绑定到纹理内存
    //torch::Tensor VolumeBackProj = torch::zeros({volHeight,volDepth,volWidth}, torch::kFloat).to(Device);
    float* VolumeBackProj_device = VolumeBackProj.data<float>();
    const cudaChannelFormatDesc channelDesc_prj = cudaCreateChannelDesc<float>();
    cudaArray* d_cube_array_proj;
    cudaMalloc3DArray(&d_cube_array_proj, &channelDesc_prj, make_cudaExtent(projWidth, projDepth, anglenums));
    cudaMemcpy3DParms copyParams_prj = { 0 };
    copyParams_prj.dstArray = d_cube_array_proj;
    copyParams_prj.extent = make_cudaExtent(projWidth, projDepth, anglenums);
    copyParams_prj.kind = cudaMemcpyDeviceToDevice;
    cudaTextureDesc     texDescr_prj;
    memset(&texDescr_prj, 0, sizeof(cudaTextureDesc));
    texDescr_prj.normalizedCoords = false;
    texDescr_prj.filterMode = cudaFilterModeLinear;
    texDescr_prj.addressMode[0] = cudaAddressModeBorder;
    texDescr_prj.addressMode[1] = cudaAddressModeBorder;
    texDescr_prj.addressMode[2] = cudaAddressModeBorder; texDescr_prj.readMode = cudaReadModeElementType;
    cudaResourceDesc    texRes_prj;
    memset(&texRes_prj, 0, sizeof(cudaResourceDesc));
    texRes_prj.resType = cudaResourceTypeArray;

    cudaTextureObject_t tex_proj;

    //注意,这里仅有两个维度
    copyParams_prj.srcPtr = make_cudaPitchedPtr(residual_device, projWidth * sizeof(float), projWidth, projDepth);
    cudaMemcpy3D(&copyParams_prj);
    texRes_prj.res.array.array = d_cube_array_proj;
    cudaCreateTextureObject(&tex_proj, &texRes_prj, &texDescr_prj, NULL);

    const dim3 blockSize_vol = dim3(BLOCK_X, BLOCK_Y);
    const dim3 gridSize_vol = dim3((volWidth + blockSize_vol.x - 1) / blockSize_vol.x, (volDepth + blockSize_vol.y - 1) / blockSize_vol.y);

    for (int angle = 0; angle < anglenums; angle = angle + 1) {
        bproj << < blockSize_vol, gridSize_vol >> > (VolumeBackProj_device, residual_device, volumeSize_device,
            projSize_device, projVector_device, volinfo_device, anglenums, tex_proj, angle);
    }
    cudaDestroyTextureObject(tex_proj);
    cudaFreeArray(d_cube_array_proj);
    return;
}



