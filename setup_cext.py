"""Build script for fast_expert_io C extension."""

from setuptools import setup, Extension
import numpy as np

fast_expert_io = Extension(
    'fast_expert_io',
    sources=['fast_expert_io.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=[
        '-O3',
        '-march=armv8.4-a',
        '-flto',
        '-DPAGE_SIZE=16384',
        '-Wno-deprecated-declarations',
    ],
    extra_link_args=[
        '-lpthread',
        '-flto',
    ],
)

setup(
    name='fast_expert_io',
    version='0.1.0',
    description='High-throughput expert weight I/O via preadv + pthreads',
    ext_modules=[fast_expert_io],
)
