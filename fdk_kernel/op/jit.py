from torch.utils.cpp_extension import load
FistaReconProj_CUDA = load('CTReconProj',['CTProjCUDA.cpp','CTProjCUDA_kernel.cu'],verbose=True)
