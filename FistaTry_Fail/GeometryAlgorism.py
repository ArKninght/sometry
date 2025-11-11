import numpy as np
import torch
from tqdm import trange
import CTFDKReconProj as FDK
from options import *
from FistaTry_Fail.ProjGemotry import getProjectionVector


volumeSize = torch.IntTensor(nVolumeSize)
detectorSize = torch.IntTensor(nDetectorSize)

parameters = torch.FloatTensor(AngleNum * 12)
getProjectionVector(parameters.numpy(), DSD, DSO, dVolumeSize, dDetectorSize, nVolumeSize, nDetectorSize, TotalAngle, AngleNum, 0)
parameters = torch.from_numpy(parameters.numpy().astype(np.float32)).contiguous()

# 定义相关的类来实现正反投影算法，确保算法在GPU上顺序运行
class ForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output):
        device = input.device
        FDK.fproj(input, output, parameters.to(device), device.index)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        device = grad.device
        grad = grad.reshape(AngleNum, detectorSize[1], volumeSize[2], detectorSize[0]).permute(0,2,1,3).reshape(1, 1, AngleNum*volumeSize[2], detectorSize[1], detectorSize[0])
        volume = FDK.bproj(grad, ctx.saved_tensors[0], parameters.to(device), device.index)
        return volume
    
class BackProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output):
        device = input.device
        input = input.reshape(AngleNum, detectorSize[1], dDetectorSize[0])
        FDK.bproj(output, input, parameters.to(device), device.index)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        device = grad.device
        FDK.fproj(ctx.saved_tensors[0], grad, parameters.to(device), device.index) * sampleInterval
        return grad
    
class DTVFista():
    def __init__(self, cascades:int=30, debug:bool=False):
        self.cascades = cascades
        self.debug = debug
        self.lamb = 0 #10e3#TV正则化系数被禁用
        self.L = 0.0001#固定步长
        #self.L = 1.0 / power_method(ForwardProjection, BackProjection)  # 自适应步长

    def run(self, image, sino, imgRes, projRes):
        t = 1
        I = Ip = image
        y = I
        for cascade in trange(self.cascades):
            imgRes.zero_()
            projRes.zero_()
            temp1 = ForwardProjection.apply(y, projRes)
            temp = temp1 - sino
            d = y - self.L * BackProjection.apply(imgRes, temp) 
            I = torch.sign(d) * torch.nn.functional.relu(torch.abs(d) - self.lamb*self.L)
            tp = (1+np.sqrt(1+4*t**2))/2
            y = I + (t-1)/tp * (I-Ip)
            Ip = I
            if cascade % 5 ==0:
                I.detach().cpu().numpy().tofile("./"+str(cascade)+".raw")
            t = tp
        return I