#!/usr/bin/env python

import os
import subprocess
import sys

from setuptools import setup, find_packages
import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension as build_ext_orig
from distutils.command.build import build as build_orig
from setuptools.command.install import install as install_orig
from setuptools.command.develop import develop as develop_orig
from distutils.errors import DistutilsFileError, DistutilsArgError
from distutils.dir_util import mkpath
from distutils.spawn import spawn

this_file = os.path.dirname(__file__)
nvvl_path = os.path.join(this_file, "../")

mycmdclass = {}
for super_class in [build_ext_orig, build_orig, install_orig, develop_orig]:
    class command(super_class):
        user_options = super_class.user_options + [
            ('with-nvvl=', None, 'Location of built nvvl library'),
            ('system-nvvl', None, 'Use the system installed nvvl library'),
        ]
        def initialize_options(self):
            super().initialize_options()
            self.with_nvvl = None
            self.system_nvvl = None

    if (super_class.__name__ == "BuildExtension"):
        name = "build_ext"
    else:
        name = super_class.__name__

    print("Overriding", name)
    mycmdclass[name] = command

def run_build(self):
    if self.with_nvvl:
        if self.system_nvvl:
            raise DistutilsArgError("system-nvvl and with-nvvl are mutually exclusive")
        libpath = os.path.join(self.with_nvvl, "libnvvl.so")
        if not os.path.isfile(libpath):
            raise DistutilsFileError("Provided with-nvvl path, but " + libpath + " doesn't exit.")
        for ext in self.extensions:
            ext.library_dirs += [self.with_nvvl]
        self.distribution.data_files = [
            ('nvvl/', [libpath])]

    elif not self.system_nvvl:
        output_dir = os.path.join(self.build_temp, "nvvl-build")
        mkpath(output_dir, 0o777, dry_run=self.dry_run)
        cmake_cmd = ["cmake", "-B"+output_dir, "-H"+nvvl_path]
        spawn(cmake_cmd, dry_run=self.dry_run)
        make_cmd = ["make", "-C", output_dir, "-j4"]
        spawn(make_cmd, dry_run=self.dry_run)



        for ext in self.extensions:
            ext.library_dirs += [output_dir]
            ext.runtime_library_dirs = ["$ORIGIN"]
        self.distribution.data_files = [
            ('nvvl/', [os.path.join(output_dir, "libnvvl.so")])]

    build_ext_orig.run(self)

def finalize_options_build(self):
    build_ext_orig.finalize_options(self)
    for cmd in ['install', 'develop', 'build']:
        self.set_undefined_options(cmd,
                                   ('system_nvvl', 'system_nvvl'),
                                   ('with_nvvl', 'with_nvvl')
        )

mycmdclass["build_ext"].run = run_build
mycmdclass["build_ext"].finalize_options = finalize_options_build

this_file = os.path.dirname(__file__)

nvvl_path = os.path.join(this_file, "../")

sources = ['src/nvvl_th.cpp']


defines = [('WITH_CUDA', None)]
include_dirs = [os.path.join(nvvl_path, "include"),
                os.path.join(nvvl_path, "src")]

nvvl_ext = CUDAExtension(
    'nvvl._nvvl',
    sources=sources,
    define_macros=defines,
    extra_objects=[],
    languages=["c++"],
    extra_compile_args=['-std=c++14'],
    include_dirs=include_dirs,
    libraries=['nvvl']
)


setup(
    name="nvvl",
    version="1.1",
    description="Read frame sequences from a video file",
    license="BSD",
    url="https://github.com/NVIDIA/nvvl",
    author="Jared Casper",
    author_email="jcasper@nvidia.com",
    setup_requires=["cmake"],
    ext_modules=[nvvl_ext],
    packages=["nvvl"],
    cmdclass=mycmdclass
)
