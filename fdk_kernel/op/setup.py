from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

setup(
    name='CTReconProj',
    ext_modules=[
        CUDAExtension('CTReconProj',[
            'CTProjCUDA.cpp',
            'CTProjCUDA_kernel.cu'
        ]),
    ],
    zip_safe = False,
    cmdclass={
        'build_ext':BuildExtension
    })