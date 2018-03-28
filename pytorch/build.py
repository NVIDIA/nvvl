import os
import torch
import subprocess
from string import Template
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

nvvl_path = os.path.join(this_file, "../")

sources = ['nvvl/src/nvvl_th.c']

headers = ['nvvl/src/nvvl_generated.h',
           'nvvl/src/nvvl_th.h']

defines = [('WITH_CUDA', None)]
include_dirs = [os.path.join(nvvl_path, "include")]

subprocess.call(["make"])

# if not torch.cuda.is_available():
#     raise RuntimeError('CUDA must be available to use this package.')

ffi = create_extension(
    'nvvl.lib',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=True,
    extra_objects=[],
    extra_compile_args=['-std=c99'],
    include_dirs=include_dirs,
    libraries=['nvvl']
)

if __name__ == '__main__':
    ffi.build()
