from torch.utils.cpp_extension import load
FistaReconProj_CUDA = load('CTFDKReconProj',['CTfdkCUDA.cpp','FDK_kernel.cu'],verbose=True)
