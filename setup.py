from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CUDAExtension

__version__ = '0.0.0'

ext_modules = []
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

if torch.cuda.is_available():
    ext_modules += [
        CUDAExtension('quat_to_mat',
                      ['cuda/quat_to_mat.cpp', 'cuda/quat_to_mat_kernels.cu']),
    ]

setup(
    name='quat_to_mat',
    version=__version__,
    setup_requires=[],
    tests_require=[],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages())
