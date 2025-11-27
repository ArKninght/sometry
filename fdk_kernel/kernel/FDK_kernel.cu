#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "helper_math.h"

#define PI 3.141592654
#define BLOCK_X 32
#define BLOCK_Y 32

// 正投影核函数
__global__ void forwardKernel(cudaTextureObject_t volumeTexture, float* sino, float* projectVector, uint3 volumeSize, uint2 detectorSize, int angleIdx)
{
	uint3 detectorIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, 
		blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);	// 获取当前计算单元对应的探测器的索引
	if (detectorIdx.x > detectorSize.x || detectorIdx.y > detectorSize.y) return;		// 判断当前计算单元是否超过探测器的索引
	// 将投影几何设定为具体的参数
	float3 sourcePos = make_float3(projectVector[angleIdx * 12], projectVector[angleIdx * 12 + 1], projectVector[angleIdx * 12 + 2]);
	float3 detectorPos = make_float3(projectVector[angleIdx * 12 + 3], projectVector[angleIdx * 12 + 4], projectVector[angleIdx * 12 + 5]);
	float3 v = make_float3(projectVector[angleIdx * 12 + 6], projectVector[angleIdx * 12 + 7], projectVector[angleIdx * 12 + 8]);
	float3 u = make_float3(projectVector[angleIdx * 12 + 9], projectVector[angleIdx * 12 + 10], projectVector[angleIdx * 12 + 11]);
	// 获取体素和探测器的中心坐标
	float3 volumeCenter = make_float3(volumeSize.x / -2.0, volumeSize.y / -2.0, volumeSize.z / -2.0);
	float2 detectorCenter = make_float2(detectorSize.x / -2.0, detectorSize.y / -2.0);

	// 正投影首先获得当前探测器对应射线的射线方程
	// 获取当前探测器索引对应探测器在空间坐标系中的坐标
	// projectorVector中已经对探测器平面进行了归一化，视探测器为1*1大小。在.mat文件中有所描述
	// detectorX，detectorY可以理解为相对于探测器中心的单个探测器的坐标
	float detectorX = detectorIdx.x + detectorCenter.x;
	float detectorY = detectorIdx.y + detectorCenter.y;
	// 在像素索引加入0.5的意义：将像素坐标从边角移动到像素的中心，更好的模拟真实情况
	// 偏移在detectorPos中有所表现，在此已经是偏移后的单个探测器在坐标系中的实际坐标
	float3 detectorPixel = detectorPos + (0.5 + detectorX) * u + (0.5 + detectorY) * v;	
	float3 rayVector = normalize(detectorPixel - sourcePos); // 归一化的射线方向向量，代表射线的方向

	// 确定射线的主轴，方便后续计算
	float alpha0, alpha1;
	float rayVectorDomainDim = fmax(fabs(rayVector.x), fmax(fabs(rayVector.z), fabs(rayVector.y)));
	// 分x,y,z三种主轴讨论, 获取射线的入射点和出射点为alpha0和alpha1
	if (rayVectorDomainDim == fabs(rayVector.x))
	{
		float volumeMinEdgePoint = volumeCenter.x;
		float volumeMaxEdgePoint = volumeSize.x + volumeCenter.x;
		alpha0 = (volumeMinEdgePoint - sourcePos.x) / rayVector.x;
		alpha1 = (volumeMaxEdgePoint - sourcePos.x) / rayVector.x;
	}
	else if (rayVectorDomainDim == fabs(rayVector.y))
	{
		float volumeMinEdgePoint = volumeCenter.y;
		float volumeMaxEdgePoint = volumeSize.y + volumeCenter.y;
		alpha0 = (volumeMinEdgePoint - sourcePos.y) / rayVector.y;
		alpha1 = (volumeMaxEdgePoint - sourcePos.y) / rayVector.y;
	}
	else
	{
		float volumeMinEdgePoint = volumeCenter.z;
		float volumeMaxEdgePoint = volumeSize.z + volumeCenter.z;
		alpha0 = (volumeMinEdgePoint - sourcePos.z) / rayVector.z;
		alpha1 = (volumeMaxEdgePoint - sourcePos.z) / rayVector.z;
	}
	// 对alpha0和alpha1进行一定程度的扩展，防止遗漏
	float minAlpha = fmin(alpha0, alpha1) - 3;
	float maxAlpha = fmax(alpha0, alpha1) + 3;

	float px, py, pz;	// 设定积分点的初始参数
	float pixel = 0;	// 设定投影积分值
	// 从初始点到终点对射线进行积分，获取模拟投影值
	while (minAlpha < maxAlpha)
	{
		// 根据alpha计算挡墙投影点，依据：p=s+alpha*rayvector
		px = sourcePos.x + minAlpha * rayVector.x;
		py = sourcePos.y + minAlpha * rayVector.y;
		pz = sourcePos.z + minAlpha * rayVector.z;
		// 将坐标转换为纹理内存的索引值
		px = px - volumeCenter.x;
		py = py - volumeCenter.y;
		pz = pz - volumeCenter.z;
		// +0.5代表将索引改为以像素中心为基准的索引
		pixel += tex3D<float>(volumeTexture, px + 0.5f, py + 0.5f, pz + 0.5f);
		minAlpha++; // 累加进行积分
	}
	// 最后对当前探测器的投影值进行赋值
	unsigned int sinoIdx = angleIdx * detectorSize.x * detectorSize.y + detectorIdx.y * detectorSize.x + detectorIdx.x;
	sino[sinoIdx] = pixel;
}

__global__ void resproForwardKernel(float* sino, float* projectVector, uint3 volumeSize, uint2 detectorSize, int angleIdx)
{
	uint3 detectorIdx = make_uint3(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);
	
	float3 sourcePos = make_float3(projectVector[12 * angleIdx], projectVector[12 * angleIdx + 1], projectVector[12 * angleIdx + 2]);
	float3 detectorPos = make_float3(projectVector[12 * angleIdx + 3], projectVector[12 * angleIdx + 4], projectVector[12 * angleIdx + 5]);
	float3 v = make_float3(projectVector[12 * angleIdx + 6], projectVector[12 * angleIdx + 7], projectVector[12 * angleIdx + 8]);
	float3 u = make_float3(projectVector[12 * angleIdx + 9], projectVector[12 * angleIdx + 10], projectVector[12 * angleIdx + 11]);

	float2 detectorCenter = make_float2(detectorSize.x / 2.0, detectorSize.y / 2.0);
	float3 volumeCenter = make_float3(volumeSize.x / 2.0, volumeSize.y / 2.0, volumeSize.z / 2.0);

	float detectorX = detectorIdx.x + detectorCenter.x;
	float detectorY = detectorIdx.y + detectorCenter.y;

	float3 detectorPixel = detectorPos + (0.5 + detectorX) * u + (0.5 + detectorY) * v;
	float3 rayVector = normalize(sourcePos - detectorPixel);

	float alpha0, alpha1;
	float mainAxis = fmax(fabs(rayVector.x), fmax(fabs(rayVector.y), fabs(rayVector.z)));
	if (mainAxis == fabs(rayVector.x))
	{
		// 计算射线经过的边界
		float volumeMinEdgePoint = volumeCenter.x;
		float volumeMaxEdgePoint = volumeSize.x + volumeCenter.x;
		// 通过边界条件计算射线与封闭盒相交的起始点与结束点的参数
		alpha0 = (volumeMinEdgePoint - sourcePos.x) / rayVector.x;
		alpha1 = (volumeMaxEdgePoint - sourcePos.x) / rayVector.x;
	}
	else if (mainAxis == fabs(rayVector.y))
	{
		float volumeMinEdgePoint = volumeCenter.y;
		float volumeMaxEdgePoint = volumeCenter.y + volumeSize.y;
		alpha0 = (volumeMinEdgePoint - sourcePos.y) / rayVector.y;
		alpha1 = (volumeMaxEdgePoint - sourcePos.y) / rayVector.y;
	}
	else
	{
		float volumeMinEdgePoint = volumeCenter.z;
		float volumeMaxEdgePoint = volumeCenter.z + volumeSize.z;
		alpha0 = (volumeMinEdgePoint - sourcePos.z) / rayVector.z;
		alpha1 = (volumeMaxEdgePoint - sourcePos.z) / rayVector.z;
	}
	float maxAlpha = fmax(alpha0, alpha1) + 3;
	float minAlpha = fmax(alpha0, alpha1) - 3;
	float len = maxAlpha - minAlpha;

	unsigned int sinoIdx = angleIdx * detectorSize.x * detectorSize.y + detectorIdx.y * detectorSize.x + detectorIdx.x;
	float pixel = sino[sinoIdx];
	pixel /= len;
	sino[sinoIdx] = pixel;
}

__global__ void backwardKernel(cudaTextureObject_t sinoTexture, float* Volume, float* projectVector, uint3 volumeSize, uint2 detectorSize, int angleIdx, int angles_num)
{
	uint3 volumeIdx = make_uint3(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);
	if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y) return;
	float3 sourcePos = make_float3(projectVector[angleIdx * 12], projectVector[angleIdx * 12 + 1], projectVector[angleIdx * 12 + 2]);
	float3 detectorPos = make_float3(projectVector[angleIdx * 12 + 3], projectVector[angleIdx * 12 + 4], projectVector[angleIdx * 12 + 5]);
	float3 v = make_float3(projectVector[angleIdx * 12 + 6], projectVector[angleIdx * 12 + 7], projectVector[angleIdx * 12 + 8]);
	float3 u = make_float3(projectVector[angleIdx * 12 + 9], projectVector[angleIdx * 12 + 10], projectVector[angleIdx * 12 +11]);
	// 获取体素和探测器的中心坐标
	float3 volumeCenter = make_float3(volumeSize.x / -2.0, volumeSize.y / -2.0, volumeSize.z / -2.0);
	float2 detectorCenter = make_float2(detectorSize.x / -2.0, detectorSize.y / -2.0);

	// 计算反投影几何，对当前角度的投影进行重建
	// 在输入中存在当前投影角度的索引，因此该核函数的主要目的是将该角度下的投影重建回原始图像中
	for (int z0 = 0; z0 < volumeSize.z; z0++)
	{
		float value = 0;
		int volumePtrIdx = z0 * volumeSize.x * volumeSize.y + volumeIdx.y * volumeSize.x + volumeIdx.x;
		float3 coordinates = make_float3(volumeIdx.x + volumeCenter.x, volumeIdx.y + volumeCenter.y, z0 + volumeCenter.z);// 当前重建点在坐标系中的坐标
		float fScale = __fdividef(1.0f, det3(u, v, sourcePos - coordinates));
		fScale = det3(u, v, sourcePos - coordinates) == 0 ? 0 : fScale;
        float detectorX = fScale * det3(coordinates - sourcePos, v, sourcePos - detectorPos) - detectorCenter.x;
        float detectorY = fScale * det3(u, coordinates - sourcePos, sourcePos - detectorPos) - detectorCenter.y;
		float fr = fScale * det3(u, v, sourcePos - detectorPos);
		if (detectorX < -1 || detectorX > detectorSize.x + 1 || detectorY < -1 || detectorY > detectorSize.y + 1) continue;
		value = fr * tex3D<float>(sinoTexture, detectorX, detectorY, angleIdx + 0.5);
		Volume[volumePtrIdx] += value * 2 * PI / angles_num;
	}
}

void forward(torch::Tensor volume, torch::Tensor sinogram ,torch::Tensor projectVector, int device)
{
    TORCH_CHECK(volume.is_cuda(), "Volume tensor must be a CUDA tensor");
    TORCH_CHECK(projectVector.is_cuda(), "Project vectors tensor must be a CUDA tensor");
    TORCH_CHECK(volume.is_contiguous(), "Volume tensor must be contiguous");
    TORCH_CHECK(projectVector.is_contiguous(), "Project vectors tensor must be contiguous");
    TORCH_CHECK(volume.scalar_type() == torch::kFloat32, "Volume tensor must be of type float32");
    TORCH_CHECK(projectVector.scalar_type() == torch::kFloat32, "Project vectors tensor must be of type float32");
    TORCH_CHECK(volume.dim() == 3, "Volume tensor must be 3D");

	// 将torch数据转换为c数据
	uint3 volumeSize = make_uint3(volume.size(2), volume.size(1), volume.size(0));
	uint2 detectorSize = make_uint2(sinogram.size(2), sinogram.size(1));
	float* volumePtr = volume.data<float>();
	float* sinoPtr = sinogram.data<float>();
	float* projectVectorPtr = projectVector.data<float>();
	int angles_num = sinogram.size(0);

	// 1. 创建通道描述符
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	// 2. 分配3D CUDA数组
	cudaExtent extent = make_cudaExtent(volumeSize.x, volumeSize.y, volumeSize.z);
	cudaArray* volumeArray;
	cudaMalloc3DArray(&volumeArray, &channelDesc, extent);
	// 3. 复制数据到CUDA数组
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(volumePtr, volumeSize.x * sizeof(float), volumeSize.x, volumeSize.y);
	copyParams.dstArray = volumeArray;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);
	// 4. 配置资源描述符
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = volumeArray;
	// 5. 配置纹理描述符
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	// 6. 创建纹理对象
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	// 以角度为单位做探测器像素驱动的正投影
	const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, 1);
	const dim3 gridSize = dim3(detectorSize.x / BLOCK_X + 1, detectorSize.y / BLOCK_Y + 1, 1);
	for (int angleIdx = 0; angleIdx < angles_num; angleIdx++) {
		forwardKernel <<<gridSize, blockSize >>> (texObj, sinoPtr, projectVectorPtr, volumeSize, detectorSize, angleIdx);
	}
	// 解绑纹理
	cudaDestroyTextureObject(texObj);  // 先销毁纹理对象
	cudaFreeArray(volumeArray);        // 再释放 CUDA Array
	return;
}

void backward(torch::Tensor volume, torch::Tensor sinogram ,torch::Tensor projectVector, int device)
{
	TORCH_CHECK(volume.is_cuda(), "Volume tensor must be a CUDA tensor");
    TORCH_CHECK(projectVector.is_cuda(), "Project vectors tensor must be a CUDA tensor");
    TORCH_CHECK(volume.is_contiguous(), "Volume tensor must be contiguous");
    TORCH_CHECK(projectVector.is_contiguous(), "Project vectors tensor must be contiguous");
    TORCH_CHECK(volume.scalar_type() == torch::kFloat32, "Volume tensor must be of type float32");
    TORCH_CHECK(projectVector.scalar_type() == torch::kFloat32, "Project vectors tensor must be of type float32");
    TORCH_CHECK(volume.dim() == 3, "Volume tensor must be 3D");

	// 将torch数据转换为c数据
	uint3 volumeSize = make_uint3(volume.size(2), volume.size(1), volume.size(0));
	uint2 detectorSize = make_uint2(sinogram.size(2), sinogram.size(1));
	float* volumePtr = volume.data<float>();
	float* sinoPtr = sinogram.data<float>();
	float* projectVectorPtr = projectVector.data<float>();
	int angles_num = sinogram.size(0);

	// 1. 创建通道描述符
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	// 2. 分配3D CUDA数组
	cudaExtent extent = make_cudaExtent(detectorSize.x, detectorSize.y, angles_num);
	cudaArray* sinoArray;
	cudaMalloc3DArray(&sinoArray, &channelDesc, extent);
	// 3. 复制数据到CUDA数组
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)sinoPtr, detectorSize.x * sizeof(float), detectorSize.x, detectorSize.y);
	copyParams.dstArray = sinoArray;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);
	// 4. 配置资源描述符
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = sinoArray;
	// 5. 配置纹理描述符
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	// 6. 创建纹理对象
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, 1);
	const dim3 gridSize = dim3(volumeSize.x / BLOCK_X + 1, volumeSize.y / BLOCK_Y + 1, 1);

	for (int angleIdx = 0; angleIdx < angles_num; angleIdx++)
	{
		backwardKernel<<<gridSize, blockSize>>>(texObj, volumePtr, projectVectorPtr,
						 volumeSize, detectorSize, angleIdx, angles_num);
	}
	// 解绑纹理
	cudaDestroyTextureObject(texObj);  // 先销毁纹理对象
	cudaFreeArray(sinoArray);        // 再释放 CUDA Array
	return;
}