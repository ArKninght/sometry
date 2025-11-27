from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

setup(
    name='CTFDKReconProj',
    ext_modules=[
        CUDAExtension('CTFDKReconProj',[
            'CTfdkCUDA.cpp',
            'FDK_kernel.cu'
        ]),
    ],
    zip_safe = False,
    cmdclass={
        'build_ext':BuildExtension
    })
