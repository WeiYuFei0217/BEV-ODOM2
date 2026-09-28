import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name, module, sources, sources_cuda=(), extra_args=(), extra_include_path=()):
    sources, extra_args = list(sources), list(extra_args)
    define_macros = []
    extra_compile_args = {'cxx': list(extra_args)}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources = sources + list(sources_cuda)
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=list(extra_include_path),
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


setup(
    name='bevodom2',
    description='BEV-ODOM2: BEV-based monocular visual odometry with PV-BEV fusion and dense flow supervision',
    url=None,
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[],
    ext_modules=[
        make_cuda_ext(
            name='voxel_pooling_train_ext',
            module='bevodom2.ops.voxel_pooling_train',
            sources=['src/voxel_pooling_train_forward.cpp'],
            sources_cuda=['src/voxel_pooling_train_forward_cuda.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
