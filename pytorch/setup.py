#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig
from distutils.command.build import build as build_orig
from setuptools.command.install import install as install_orig
from setuptools.command.develop import develop as develop_orig
from distutils.errors import DistutilsFileError, DistutilsArgError
from distutils.dir_util import mkpath
from distutils.spawn import spawn

import build

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

    mycmdclass[super_class.__name__] = command

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
            ('nvvl/lib', [libpath])]

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
            ('nvvl/lib', [os.path.join(output_dir, "libnvvl.so")])]

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

setup(
    name="nvvl",
    version="1.0",
    description="Read frame sequences from a video file",
    license="BSD",
    url="https://github.com/NVIDIA/nvvl",
    author="Jared Casper",
    author_email="jcasper@nvidia.com",
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cmake", "cffi>=1.0.0"],
    packages=find_packages(exclude=["build"]),
    ext_package="",
    cffi_modules=[
        os.path.join(this_file, "build.py:ffi")
    ],
    cmdclass=mycmdclass
)
