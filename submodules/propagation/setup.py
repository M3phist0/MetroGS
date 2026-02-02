from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='propagation',
    ext_modules=[
        CUDAExtension('propagation', 
            sources=[
                'PatchMatch.cpp', 
                'Propagation.cu',
                'pro.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3',
                    '-D_GLIBCXX_USE_CXX11_ABI=1',
                    '-gencode=arch=compute_120,code=sm_120',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
