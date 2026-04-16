from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_flash_attn',
    ext_modules=[
        CUDAExtension(
            name='custom_flash_attn',
            sources=[
                'csrc/flash_attn_api.cpp',
                'csrc/kernels/flash_attn_fp32.cu',
                'csrc/kernels/flash_attn_fp16.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3', 
                    '-arch=sm_75', 
                    '--allow-unsupported-compiler'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)